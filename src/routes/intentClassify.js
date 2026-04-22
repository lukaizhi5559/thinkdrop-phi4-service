/**
 * POST /intent.classify
 * Local zero-shot intent classifier using @xenova/transformers (NLI-DeBERTa-v3-small).
 * Used by parseIntent.js ensemble adjudication to cross-check the LLM's estimatedIntent.
 *
 * Payload:  { message: string, llmIntent?: string, llmConfidence?: number }
 * Response: { topIntent, topConfidence, scores{}, top2Gap, method, elapsedMs }
 *
 * Reuses the same Xenova pipeline singleton as domain.js — both share
 * Xenova/nli-deberta-v3-small. Pipeline is lazy-loaded on first request.
 */
const express = require('express');
const router = express.Router();

// ── Singleton pipeline shared with domain.js ──────────────────────────────────
// Declared module-level so the pipeline is loaded at most once per process.
let _zeroShotPipeline = null;
async function getZeroShotPipeline() {
  if (!_zeroShotPipeline) {
    const { pipeline } = require('@xenova/transformers');
    _zeroShotPipeline = await pipeline('zero-shot-classification', 'Xenova/nli-deberta-v3-small');
  }
  return _zeroShotPipeline;
}

// ── Intent label vocabulary ───────────────────────────────────────────────────
// Ordered from most specific to least specific for NLI scoring stability.
const INTENT_LABELS = [
  'command_automate',    // browser navigation, form fill, click, send — any app/web action
  'app_control_start',   // launching or controlling a local app (open Slack, control mode)
  'screen_intelligence', // read or describe what is on the user's screen right now
  'web_search',          // factual lookup without navigating a specific website
  'memory_store',        // save, remember, note, record a personal fact
  'memory_retrieve',     // recall, what did i say, look up stored personal info
  'general_knowledge',   // general Q&A, explain, summarise — no action, no memory
  'greeting',            // greetings, small talk, pleasantries
];

// ── Natural-language descriptions fed to NLI model ───────────────────────────
// DeBERTa-NLI expects a hypothesis that sounds natural, not just a label word.
const LABEL_HYPOTHESES = {
  command_automate:    'The user wants to automate a task, navigate to a website, open an app, fill a form, or perform an action on the computer.',
  app_control_start:   'The user wants to launch, start, open, or take control of a specific application on their device.',
  screen_intelligence: 'The user wants to know what is currently visible on their screen or wants the AI to read what is shown.',
  web_search:          'The user is asking a factual question or wants information looked up without navigating to a specific website.',
  memory_store:        'The user wants to save, remember, note down, or store a personal fact or piece of information.',
  memory_retrieve:     'The user wants to recall, retrieve, or find something they previously told the assistant.',
  general_knowledge:   'The user is asking a general question, wants an explanation, or wants a summary with no action required.',
  greeting:            'The user is greeting, saying hello, or making small talk.',
};

router.post('/intent.classify', async (req, res) => {
  const startTime = Date.now();
  const { requestId, payload } = req.body;
  const { message, llmIntent, llmConfidence } = payload || {};

  if (!message) {
    return res.status(400).json({
      version: 'mcp.v1', service: 'phi4', action: 'intent.classify', requestId,
      status: 'error',
      error: { code: 'INVALID_REQUEST', message: 'payload.message is required', retryable: false },
    });
  }

  try {
    const scores = {};
    let topIntent = 'general_knowledge';
    let topConfidence = 0;
    let method = 'zero-shot';

    // ── Zero-shot classification ──────────────────────────────────────────────
    let zeroShotFailed = false;
    try {
      const classifier = await getZeroShotPipeline();
      // Use per-label hypothesis text for better NLI entailment scoring.
      const hypotheses = INTENT_LABELS.map(l => LABEL_HYPOTHESES[l]);
      const result = await classifier(message.substring(0, 512), hypotheses, { multi_label: false });

      // result.labels are the hypothesis strings; map back to intent label names
      result.labels.forEach((hypothesis, i) => {
        const label = INTENT_LABELS[hypotheses.indexOf(hypothesis)];
        if (label) scores[label] = result.scores[i];
      });

      // Find top intent
      let best = -1;
      for (const [label, score] of Object.entries(scores)) {
        if (score > best) { best = score; topIntent = label; topConfidence = score; }
      }
    } catch (err) {
      console.warn('[phi4/intent.classify] Zero-shot failed:', err.message);
      zeroShotFailed = true;
      method = 'keyword-fallback';
    }

    // ── Keyword fallback ──────────────────────────────────────────────────────
    if (zeroShotFailed) {
      const lowerMsg = message.toLowerCase();
      const ACTION_VERBS = /\b(go\s+to|goto|navigate|open|click|fill|send|submit|search\s+on|look\s+up\s+on|find\s+on|use|visit|run|execute|deploy|book|schedule|reply|compose|draft|text|tweet|post\s+on)\b/i;
      const MEMORY_STORE  = /\b(remember|save|note|store|don.?t forget|keep track)\b/i;
      const MEMORY_GET    = /\b(what did i|recall|what.?s my|do you remember|i told you)\b/i;
      const SCREEN        = /\b(what.?s on|what do you see|read the screen|screen shows|current screen)\b/i;
      const GREETING_RE   = /^(hi|hello|hey|good\s+(morning|evening|afternoon)|what.?s up|howdy|yo)\b/i;

      if (GREETING_RE.test(lowerMsg))       { topIntent = 'greeting';           topConfidence = 0.80; }
      else if (SCREEN.test(lowerMsg))       { topIntent = 'screen_intelligence'; topConfidence = 0.75; }
      else if (MEMORY_STORE.test(lowerMsg)) { topIntent = 'memory_store';        topConfidence = 0.75; }
      else if (MEMORY_GET.test(lowerMsg))   { topIntent = 'memory_retrieve';     topConfidence = 0.75; }
      else if (ACTION_VERBS.test(lowerMsg)) { topIntent = 'command_automate';    topConfidence = 0.70; }
      else                                  { topIntent = 'general_knowledge';   topConfidence = 0.60; }

      scores[topIntent] = topConfidence;
    }

    // ── top2Gap: confidence margin between #1 and #2 ─────────────────────────
    const sortedScores = Object.entries(scores).sort((a, b) => b[1] - a[1]);
    const top2Gap = sortedScores.length >= 2
      ? sortedScores[0][1] - sortedScores[1][1]
      : 1.0;

    const elapsedMs = Date.now() - startTime;
    console.log(`[phi4/intent.classify] "${message.slice(0, 60)}" → ${topIntent} (${topConfidence.toFixed(3)}) gap=${top2Gap.toFixed(3)} llmIntent=${llmIntent || 'none'} ${elapsedMs}ms`);

    res.json({
      version: 'mcp.v1',
      service: 'phi4',
      action: 'intent.classify',
      requestId,
      status: 'ok',
      data: {
        topIntent,
        topConfidence,
        scores,
        top2Gap,
        method,
        elapsedMs,
      },
      error: null,
      metrics: { elapsedMs },
    });
  } catch (err) {
    console.error('[phi4/intent.classify] Unexpected error:', err.message);
    res.status(500).json({
      version: 'mcp.v1', service: 'phi4', action: 'intent.classify', requestId,
      status: 'error',
      error: { code: 'INTERNAL_ERROR', message: err.message, retryable: true },
    });
  }
});

module.exports = router;
