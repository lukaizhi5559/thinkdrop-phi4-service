/**
 * Decision Routes for Phi4 Service
 *
 * POST /decision.answer — typed bounded decisions via a local Ollama
 * System One decision model (POST {OLLAMA_HOST}/v1/systemone).
 *
 * Replaces remote `taskType:'classification'` LLM calls for fixed-vocabulary
 * questions: lane picks, yes/no flags, ratings. One batched call answers every
 * question in a single forward pass (~50–300ms warm on Apple Silicon vs
 * 1.6–12.5s remote, with observed 900s+ pathological tails).
 *
 * IMPORTANT: System One models CANNOT generate text. Only three question
 * types exist: choice (2–26 named options), noul (yes/no → probability of
 * true), score (0..N-1 weighted level). There is no cloze/match/timeframe —
 * callers needing text must use a generative backend.
 *
 * Model tiers adapt to host RAM at boot (DECISION_MODEL overrides):
 *   <12GB  → tev1:0.8b   (~0.8GB, text-only, fastest)
 *   12–24GB→ d3:2b       (~3.6GB, vision-capable)
 *   ≥24GB  → nimble      (~5.5GB, highest quality)
 * Requests carrying an image use DECISION_VISION_MODEL (default d3:2b)
 * regardless of tier — tev1/laya are text-only.
 *
 * External contract (unchanged from the original shim):
 *   Request  payload: { state?, questions:[{id,type,question,options?,min?,max?}], image?, options? }
 *   Response data:    { answers: { <id>: {index?,label?,bool?,probability?,number?,confidence?,probabilities?} }, failed[], model, elapsedMs }
 * Fail-open: Ollama down/model missing → status:'error', code
 * 'DECISION_UNAVAILABLE', retryable:true — callers fall back to remote.
 */

const express = require('express');
const os = require('os');
const router = express.Router();

const OLLAMA_HOST = process.env.OLLAMA_HOST || 'http://localhost:11434';
const REQUEST_TIMEOUT_MS = parseInt(process.env.DECISION_REQUEST_TIMEOUT_MS || '30000', 10);

// ── Adaptive model tier ────────────────────────────────────────────────────
function _pickTier() {
  const gb = os.totalmem() / (1024 ** 3);
  if (gb < 12) return { tier: 'minimal', model: process.env.DECISION_MODEL_SMALL || 'tev1:0.8b' };
  if (gb < 24) return { tier: 'standard', model: process.env.DECISION_MODEL_MID || 'd3:2b' };
  return { tier: 'full', model: process.env.DECISION_MODEL_LARGE || 'nimble:latest' };
}
const _tier = _pickTier();
const DECISION_MODEL = process.env.DECISION_MODEL || _tier.model;
const DECISION_VISION_MODEL = process.env.DECISION_VISION_MODEL || 'd3:2b';
// The vision model is only invoked on judged screen captures — pinning it for
// 10m keeps ~1.3GB RSS resident on small-RAM machines even when idle. A short
// keep_alive lets Ollama evict it between captures; a cold judge pays the
// reload once instead of holding RAM hostage.
const VISION_KEEP_ALIVE = process.env.OLLAMA_VISION_KEEP_ALIVE || '90s';
console.log(`[phi4/decision] tier=${_tier.tier} ram=${(os.totalmem() / 1024 ** 3).toFixed(1)}GB model=${DECISION_MODEL} vision=${DECISION_VISION_MODEL}`);

// Ollama reachability probe — cached 30s, same pattern the old llm.js used.
let _ollamaAvailable = null;
let _ollamaCheckedAt = 0;
async function isOllamaAvailable() {
  const now = Date.now();
  if (_ollamaAvailable !== null && now - _ollamaCheckedAt < 30000) return _ollamaAvailable;
  try {
    const net = require('net');
    const url = new URL(OLLAMA_HOST);
    await new Promise((resolve, reject) => {
      const socket = net.createConnection({ host: url.hostname || '127.0.0.1', port: parseInt(url.port) || 11434 });
      const timer = setTimeout(() => { socket.destroy(); reject(new Error('connect timeout')); }, 1500);
      socket.on('connect', () => { clearTimeout(timer); socket.destroy(); resolve(); });
      socket.on('error', (err) => { clearTimeout(timer); reject(err); });
    });
    _ollamaAvailable = true;
  } catch {
    _ollamaAvailable = false;
  }
  _ollamaCheckedAt = now;
  return _ollamaAvailable;
}

const VALID_TYPES = new Set(['choice', 'noul', 'rating', 'score']);

function _validateQuestions(questions) {
  if (!Array.isArray(questions) || questions.length === 0) return 'payload.questions must be a non-empty array';
  if (questions.length > 64) return 'payload.questions is capped at 64 per call';
  for (let i = 0; i < questions.length; i++) {
    const q = questions[i];
    if (!q || typeof q !== 'object') return `questions[${i}] must be an object`;
    if (!q.id || typeof q.id !== 'string' || !/^[\w.-]{1,100}$/.test(q.id)) {
      return `questions[${i}].id must be a 1–100 char [\\w.-] string`;
    }
    if (!VALID_TYPES.has(q.type)) {
      return `questions[${i}].type "${q.type}" unsupported — systemone only does choice/noul/rating/score (no text generation)`;
    }
    if (!q.question || typeof q.question !== 'string') return `questions[${i}].question is required`;
    if (q.type === 'choice') {
      if (!Array.isArray(q.options) || q.options.length < 2 || q.options.length > 26) {
        return `questions[${i}] choice requires 2–26 options`;
      }
      const keys = q.options.map(String);
      if (new Set(keys).size !== keys.length) return `questions[${i}] choice options must be unique`;
    }
  }
  return null;
}

/**
 * Translate our question wire format → SystemOne `questions` object.
 *   choice → { type:'choice', instructions, criteria: {optLabel: optLabel} }
 *   noul   → { type:'noul',  instructions, criteria: {true:...,false:...}? }
 *   score  → { type:'score', instructions, criteria: [levelDesc low→high] }
 */
function _toSystemOneQuestions(questions) {
  const out = {};
  for (const q of questions) {
    if (q.type === 'choice') {
      const criteria = {};
      for (const opt of q.options) criteria[String(opt)] = String(opt);
      out[q.id] = { type: 'choice', instructions: q.question, criteria };
    } else if (q.type === 'noul') {
      out[q.id] = {
        type: 'noul',
        instructions: q.question,
        ...(q.trueDesc || q.falseDesc ? { criteria: { true: q.trueDesc || 'yes', false: q.falseDesc || 'no' } } : {}),
      };
    } else { // rating | score — level descriptions, lowest→highest, ≤10
      const min = q.min ?? 0;
      let max = q.max ?? 1;
      if (max - min + 1 > 10) max = min + 9;
      const criteria = [];
      for (let v = min; v <= max; v++) criteria.push(`${v}`);
      out[q.id] = { type: 'score', instructions: q.question, criteria };
    }
  }
  return out;
}

/** Normalize a SystemOne answer back into the caller-facing shape. */
function _normalize(q, ans) {
  if (!ans || typeof ans !== 'object') return { ok: false };
  switch (q.type) {
    case 'choice': {
      const label = ans.choice;
      if (typeof label !== 'string') return { ok: false };
      const index = q.options.findIndex(o => String(o) === label);
      if (index < 0) return { ok: false };
      return { ok: true, value: { index, label, probabilities: ans.probabilities, confidence: ans.confidence } };
    }
    case 'noul': {
      const p = ans.noul;
      if (typeof p !== 'number' || !Number.isFinite(p)) return { ok: false };
      return { ok: true, value: { bool: p >= 0.5, probability: p } };
    }
    case 'rating':
    case 'score': {
      const s = ans.score;
      if (typeof s !== 'number' || !Number.isFinite(s)) return { ok: false };
      const min = q.min ?? 0;
      const max = q.max ?? 1;
      const span = Math.min(max - min, 9);
      const number = Math.min(Math.max(min + s, min), max);
      return { ok: true, value: { number, min, max, legend: ans.legend, probabilities: ans.probabilities, confidence: ans.confidence, _span: span } };
    }
    default:
      return { ok: false };
  }
}

/**
 * Vision-model fallback — general Ollama vision models (moondream, llava) are
 * not registered with /v1/systemone (llama.cpp decision models only). Small
 * VQA models also cannot follow nested-JSON answer contracts, so each question
 * is asked as a plain lettered multiple-choice / yes-no / numeric prompt via
 * /api/generate, and the parsed result is run through _normalize for the
 * identical caller-facing shape.
 */
const _LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

function _visionPrompt(q, state) {
  const ctx = String(state || '(none)').slice(0, 400);
  if (q.type === 'choice') {
    const opts = q.options.map((o, i) => `${_LETTERS[i]}. ${o}`).join('\n');
    return `Look at this screenshot of a web page.\n\nContext: ${ctx}\n\n${q.question}\n${opts}\n\nAnswer with one letter only: ${q.options.map((_, i) => _LETTERS[i]).join(', ')}.`;
  }
  if (q.type === 'noul') {
    return `Look at this screenshot of a web page.\n\nContext: ${ctx}\n\n${q.question}\n\nAnswer yes or no.`;
  }
  const min = q.min ?? 0, max = q.max ?? 1;
  return `Look at this screenshot of a web page.\n\nContext: ${ctx}\n\n${q.question}\n\nAnswer with a number from ${min} to ${max}.`;
}

function _visionParse(q, text) {
  const t = String(text || '').trim();
  if (q.type === 'choice') {
    const m = t.match(/\b([A-Za-z])\b/);
    if (m) {
      const idx = m[1].toUpperCase().charCodeAt(0) - 65;
      if (idx >= 0 && idx < q.options.length) return { choice: String(q.options[idx]) };
    }
    // Fallback: model echoed an option's lane token
    const tok = t.toLowerCase();
    const hit = q.options.find(o => {
      const lane = String(o).toLowerCase().split(/\s+/)[0];
      return lane.length > 2 && tok.includes(lane);
    });
    if (hit) return { choice: String(hit) };
    return null;
  }
  if (q.type === 'noul') {
    if (/^y(es)?\b/i.test(t)) return { noul: 1.0 };
    if (/^no\b/i.test(t)) return { noul: 0.0 };
    return null;
  }
  const n = t.match(/-?\d+/);
  if (!n) return null;
  const min = q.min ?? 0;
  return { score: Math.min(Math.max(parseInt(n[0], 10) - min, 0), 9) };
}

async function _visionGenerateFallback({ model, state, questions, image }) {
  const answers = {};
  let lastErr = null;
  for (const q of questions) {
    try {
      const resp = await fetch(`${OLLAMA_HOST}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model, prompt: _visionPrompt(q, state), images: [image], stream: false,
          keep_alive: VISION_KEEP_ALIVE, options: { temperature: 0 },
        }),
        signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
      });
      if (!resp.ok) { lastErr = `generate HTTP ${resp.status}`; continue; }
      const data = await resp.json().catch(() => null);
      const raw = _visionParse(q, data?.response);
      const n = _normalize(q, raw);
      if (n.ok) answers[q.id] = n.value;
      else lastErr = `unparseable answer for "${q.id}": ${String(data?.response || '').slice(0, 80)}`;
    } catch (e) {
      lastErr = e.message;
    }
  }
  return Object.keys(answers).length ? { ok: true, answers } : { ok: false, error: lastErr || 'vision model produced no usable answers' };
}

/**
 * POST /decision.answer
 * Payload: { state?, questions: [...], image? (base64), options?: { model? } }
 */
router.post('/decision.answer', async (req, res) => {
  const startTime = Date.now();
  const { requestId, payload } = req.body;
  const { state = '', questions, image, options = {} } = payload || {};

  const bad = _validateQuestions(questions);
  if (bad) {
    return res.status(400).json({
      version: 'mcp.v1', service: 'phi4', action: 'decision.answer', requestId,
      status: 'error', error: { code: 'INVALID_REQUEST', message: bad, retryable: false },
    });
  }

  const model = image ? DECISION_VISION_MODEL : (options.model || DECISION_MODEL);

  try {
    if (!(await isOllamaAvailable())) {
      return res.json({
        version: 'mcp.v1', service: 'phi4', action: 'decision.answer', requestId,
        status: 'error',
        error: { code: 'DECISION_UNAVAILABLE', message: 'Ollama not reachable — caller should use remote fallback', retryable: true },
        metrics: { elapsedMs: Date.now() - startTime },
      });
    }

    const body = {
      model,
      state: typeof state === 'string' ? (state || '(none)') : state,
      questions: _toSystemOneQuestions(questions),
      keep_alive: image ? VISION_KEEP_ALIVE : '10m',
      ...(image ? { images: [image] } : {}),
    };

    const resp = await fetch(`${OLLAMA_HOST}/v1/systemone`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });

    if (!resp.ok) {
      const text = await resp.text().catch(() => '');
      // Vision fallback — systemone only accepts registered llama.cpp decision
      // models (tev1:*); a general vision model (moondream, llava, …) answers
      // the same typed contract through /api/generate instead.
      if (image && /not a llama\.cpp decision model|not found|does not exist/i.test(text)) {
        const gen = await _visionGenerateFallback({ model, state, questions, image });
        if (gen.ok) {
          const elapsedMs = Date.now() - startTime;
          return res.json({
            version: 'mcp.v1', service: 'phi4', action: 'decision.answer', requestId,
            status: 'ok',
            data: { answers: gen.answers, model },
            metrics: { elapsedMs },
          });
        }
        return res.json({
          version: 'mcp.v1', service: 'phi4', action: 'decision.answer', requestId,
          status: 'error',
          error: { code: 'DECISION_MALFORMED', message: gen.error || 'vision fallback failed', retryable: true },
          metrics: { elapsedMs: Date.now() - startTime },
        });
      }
      const notFound = resp.status === 404 || /not found|does not exist/i.test(text);
      const msg = notFound
        ? `decision model "${model}" not pulled — run: ollama pull ${model}`
        : `systemone HTTP ${resp.status}: ${text.slice(0, 300)}`;
      return res.json({
        version: 'mcp.v1', service: 'phi4', action: 'decision.answer', requestId,
        status: 'error',
        error: { code: 'DECISION_UNAVAILABLE', message: msg, retryable: true },
        metrics: { elapsedMs: Date.now() - startTime },
      });
    }

    const data = await resp.json();
    const answers = {};
    const failed = [];
    for (const q of questions) {
      const r = _normalize(q, data?.answers?.[q.id]);
      if (r.ok) answers[q.id] = r.value;
      else failed.push(q.id);
    }

    const elapsedMs = Date.now() - startTime;
    if (elapsedMs > 800) {
      console.warn(`[phi4/decision] slow decision ${elapsedMs}ms model=${model} questions=${questions.length}${image ? ' (image)' : ''}`);
    }
    if (failed.length === questions.length) {
      return res.json({
        version: 'mcp.v1', service: 'phi4', action: 'decision.answer', requestId,
        status: 'error',
        error: { code: 'DECISION_MALFORMED', message: `model output unusable for all ${failed.length} question(s)`, retryable: true },
        metrics: { elapsedMs },
      });
    }

    res.json({
      version: 'mcp.v1', service: 'phi4', action: 'decision.answer', requestId,
      status: 'ok',
      data: { answers, failed, model, elapsedMs },
      error: null,
      metrics: { elapsedMs },
    });
  } catch (err) {
    const msg = err.name === 'TimeoutError' || err.name === 'AbortError'
      ? `systemone request timed out after ${REQUEST_TIMEOUT_MS}ms`
      : err.message;
    console.error('[phi4/decision.answer] Error:', msg);
    res.json({
      version: 'mcp.v1', service: 'phi4', action: 'decision.answer', requestId,
      status: 'error',
      error: { code: 'DECISION_UNAVAILABLE', message: msg, retryable: true },
      metrics: { elapsedMs: Date.now() - startTime },
    });
  }
});

module.exports = router;
