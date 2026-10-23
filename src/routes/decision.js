/**
 * Decision Routes for Phi4 Service
 *
 * POST /decision.answer — typed bounded decisions via a local Ollama model.
 *
 * Replaces remote `taskType:'classification'` LLM calls for fixed-vocabulary
 * questions: lane picks, yes/no flags, ratings, short fields. One batched
 * call answers every question in a single forward pass (~50–300ms warm on
 * Apple Silicon vs 1.6–12.5s remote, with observed 900s+ pathological tails).
 *
 * Model tiers adapt to host RAM at boot (DECISION_MODEL overrides):
 *   <12GB  → laya / tev1:0.8b  (~0.5–1GB, text-only, fastest)
 *   12–24GB→ d3:2b             (~1.3GB, vision-capable)
 *   ≥24GB  → nimble            (~5.5GB, highest quality)
 * Requests carrying an image use DECISION_VISION_MODEL (default d3:2b)
 * regardless of tier — tev1/laya are text-only.
 *
 * Question shapes (all batched into one inference):
 *   { id, type:'choice',    question, options:[..] }           → { index, label }
 *   { id, type:'noul',      question }                          → { bool }
 *   { id, type:'rating'|'score', question, min?, max? }         → { number }
 *   { id, type:'cloze'|'match'|'timeframe', question }          → { text }
 *
 * Response data: { answers: { <id>: {value, ...}, ... }, model, elapsedMs }
 * Fail-open: Ollama down/model missing → status:'error', code
 * 'DECISION_UNAVAILABLE', retryable:true — callers fall back to remote.
 */

const express = require('express');
const os = require('os');
const router = express.Router();

// Lazy-load Ollama — service must boot even when Ollama isn't installed.
let _ollama = null;
function getOllama() {
  if (!_ollama) {
    const { Ollama } = require('ollama');
    _ollama = new Ollama({ host: process.env.OLLAMA_HOST || 'http://localhost:11434' });
  }
  return _ollama;
}

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
console.log(`[phi4/decision] tier=${_tier.tier} ram=${(os.totalmem() / 1024 ** 3).toFixed(1)}GB model=${DECISION_MODEL} vision=${DECISION_VISION_MODEL}`);

// Ollama reachability probe — cached 30s, same pattern the old llm.js used.
let _ollamaAvailable = null;
let _ollamaCheckedAt = 0;
async function isOllamaAvailable() {
  const now = Date.now();
  if (_ollamaAvailable !== null && now - _ollamaCheckedAt < 30000) return _ollamaAvailable;
  try {
    const net = require('net');
    const url = new URL(process.env.OLLAMA_HOST || 'http://localhost:11434');
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

const VALID_TYPES = new Set(['choice', 'noul', 'rating', 'score', 'match', 'cloze', 'timeframe']);

function _validateQuestions(questions) {
  if (!Array.isArray(questions) || questions.length === 0) return 'payload.questions must be a non-empty array';
  if (questions.length > 30) return 'payload.questions is capped at 30 per call';
  for (let i = 0; i < questions.length; i++) {
    const q = questions[i];
    if (!q || typeof q !== 'object') return `questions[${i}] must be an object`;
    if (!q.id || typeof q.id !== 'string') return `questions[${i}].id is required (string)`;
    if (!VALID_TYPES.has(q.type)) return `questions[${i}].type must be one of ${[...VALID_TYPES].join(',')}`;
    if (!q.question || typeof q.question !== 'string') return `questions[${i}].question is required`;
    if (q.type === 'choice' && (!Array.isArray(q.options) || q.options.length < 2)) {
      return `questions[${i}] choice requires options[] with ≥2 entries`;
    }
  }
  return null;
}

/** Build the strict-output prompt. One JSON object: { "<id>": <raw answer> }. */
function _buildPrompt(state, questions) {
  const spec = questions.map((q, i) => {
    const n = i + 1;
    switch (q.type) {
      case 'choice':
        return `${n}. "${q.id}" (choice) ${q.question}\n   Options: ${q.options.map((o, j) => `${j}) ${o}`).join(' | ')}\n   Answer with the option INDEX (integer).`;
      case 'noul':
        return `${n}. "${q.id}" (yes/no) ${q.question}\n   Answer true or false.`;
      case 'rating':
      case 'score':
        return `${n}. "${q.id}" (score) ${q.question}\n   Answer a number ${q.min ?? 0}–${q.max ?? 1}.`;
      case 'match':
        return `${n}. "${q.id}" (match) ${q.question}\n   Answer with the single best matching string.`;
      case 'timeframe':
        return `${n}. "${q.id}" (timeframe) ${q.question}\n   Answer an ISO-8601 duration or a short phrase like "today", "this week".`;
      case 'cloze':
      default:
        return `${n}. "${q.id}" ${q.question}\n   Answer with a short string.`;
    }
  }).join('\n');
  return `You are a decision engine. Answer EVERY numbered question with a value of the requested type.
Return ONLY a JSON object mapping each question id to its answer. No prose, no markdown.

Context:
${state || '(none)'}

Questions:
${spec}

JSON object:`;
}

/** Normalize a raw model answer into the typed shape for its question. */
function _normalize(q, raw) {
  switch (q.type) {
    case 'choice': {
      let index = null;
      if (typeof raw === 'number' && Number.isInteger(raw)) index = raw;
      else if (typeof raw === 'string') {
        const m = raw.match(/-?\d+/);
        if (m) index = parseInt(m[0], 10);
        else {
          const li = q.options.findIndex(o => String(o).toLowerCase() === raw.toLowerCase());
          if (li >= 0) index = li;
        }
      }
      if (index == null || index < 0 || index >= q.options.length) return { ok: false };
      return { ok: true, value: { index, label: q.options[index] } };
    }
    case 'noul': {
      if (typeof raw === 'boolean') return { ok: true, value: { bool: raw } };
      const s = String(raw).trim().toLowerCase();
      if (/^(true|yes|y|1)$/.test(s)) return { ok: true, value: { bool: true } };
      if (/^(false|no|n|0)$/.test(s)) return { ok: true, value: { bool: false } };
      return { ok: false };
    }
    case 'rating':
    case 'score': {
      const n = typeof raw === 'number' ? raw : parseFloat(String(raw).match(/-?\d+(?:\.\d+)?/)?.[0]);
      if (!Number.isFinite(n)) return { ok: false };
      const min = q.min ?? 0, max = q.max ?? 1;
      return { ok: true, value: { number: Math.min(Math.max(n, min), max), min, max } };
    }
    case 'match': {
      const s = String(raw ?? '').trim();
      if (!s) return { ok: false };
      const li = Array.isArray(q.options) ? q.options.findIndex(o => String(o).toLowerCase() === s.toLowerCase()) : -1;
      return { ok: true, value: { text: li >= 0 ? q.options[li] : s, index: li >= 0 ? li : undefined } };
    }
    case 'timeframe':
    case 'cloze':
    default: {
      const s = String(raw ?? '').trim();
      if (!s) return { ok: false };
      return { ok: true, value: { text: s.slice(0, 280) } };
    }
  }
}

/**
 * POST /decision.answer
 * Payload: { state?, questions: [...], image? (base64), options?: { model?, temperature? } }
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

    const message = { role: 'user', content: _buildPrompt(state, questions) };
    if (image) message.images = [image];

    const response = await getOllama().chat({
      model,
      messages: [message],
      stream: false,
      format: 'json',
      keep_alive: '10m',
      options: { temperature: options.temperature ?? 0, num_predict: 256 },
    });

    let parsed = {};
    try {
      parsed = JSON.parse(response.message?.content || '{}');
    } catch {
      const m = String(response.message?.content || '').match(/\{[\s\S]*\}/);
      if (m) { try { parsed = JSON.parse(m[0]); } catch { /* leave {} */ } }
    }

    const answers = {};
    const failed = [];
    for (const q of questions) {
      const r = _normalize(q, parsed[q.id]);
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
    const msg = /model .* not found|does not exist/i.test(err.message)
      ? `decision model "${model}" not pulled — run: ollama pull ${model}`
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
