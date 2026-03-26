/**
 * DistilBERT Intent Parser — Fine-tuned classifier edition
 *
 * Replaces seed cosine-similarity with a proper trained classifier.
 * Model: distilbert-base-uncased fine-tuned on ~6,250 ThinkDrop intent examples.
 *
 * To train the model:
 *   python3 scripts/train-intent-classifier.py
 *
 * Architecture:
 *   OLD: all-MiniLM-L6-v2 embeddings + cosine similarity to ~5,000 seed examples
 *        Problem: cold-starts on every new proper noun (75-80% baseline per round)
 *   NEW: distilbert-base-uncased with classification head, fine-tuned on labeled data
 *        Generalises to new nouns via pattern learning (expected baseline: 90%+)
 */

const path = require('path');
const { pipeline, env } = require('@xenova/transformers');
const IntentResponses = require('../utils/IntentResponses.cjs');
const nlp = require('compromise');

// Tell @xenova/transformers to look for models locally first.
// NOTE: @xenova/transformers v2 prepends localModelPath to the model identifier
//       and appends /onnx/model_quantized.onnx (or /onnx/model.onnx).
//       So use 'intent-classifier' (without /onnx) as the identifier.
env.localModelPath = path.join(__dirname, '../../models');
env.allowRemoteModels = false; // only use local fine-tuned model

const INTENT_LABELS = [
  'command_automate',
  'app_control_start',
  'screen_intelligence',
  'web_search',
  'memory_store',
  'memory_retrieve',
  'general_knowledge',
  'greeting',
];

// Model identifier relative to localModelPath.
// @xenova/transformers looks for: <localModelPath>/<id>/onnx/model_quantized.onnx (preferred)
// or <localModelPath>/<id>/onnx/model.onnx (fallback).
const MODEL_DIR = (() => {
  const fs = require('fs');
  const baseDir = path.join(__dirname, '../../models');
  // Check for model_quantized.onnx first (preferred by @xenova/transformers)
  if (fs.existsSync(path.join(baseDir, 'intent-classifier/onnx/model_quantized.onnx'))) return 'intent-classifier';
  if (fs.existsSync(path.join(baseDir, 'intent-classifier/onnx/model.onnx'))) return 'intent-classifier';
  return null; // model not yet trained
})();

class DistilBertIntentParser {
  constructor() {
    this.classifier  = null;
    this.initialized = false;
    this.modelTrained = Boolean(MODEL_DIR);
  }

  async initialize() {
    if (this.initialized) return;

    const startTime = Date.now();
    console.log('🚀 Initializing DistilBertIntentParser (fine-tuned)...');

    if (!this.modelTrained) {
      console.error('❌ Fine-tuned model not found. Run: python3 scripts/train-intent-classifier.py');
      throw new Error('Fine-tuned intent model not found. See scripts/train-intent-classifier.py');
    }

    try {
      console.log(`  Loading model from: ${MODEL_DIR}`);
      this.classifier = await pipeline(
        'text-classification',
        MODEL_DIR,
        { topk: null }  // return scores for ALL classes, not just top-1
      );

      this.initialized = true;
      console.log(`✅ DistilBertIntentParser initialized in ${Date.now() - startTime}ms`);
    } catch (error) {
      console.error('❌ Failed to initialize DistilBertIntentParser:', error);
      throw error;
    }
  }

  /**
   * Classify intent from a text message.
   * Returns the same shape as the old parser so callers don't need changes.
   */
  async parse(message, options = {}) {
    if (!this.initialized) await this.initialize();

    const startTime = Date.now();

    try {
      // Strip highlighted-text markers (used by screen_intelligence logic upstream)
      const hasHighlight = message.includes('[HIGHLIGHTED_TEXT]');
      let messageToClassify = message.replace(/\[HIGHLIGHTED_TEXT\]\s*/g, '').trim();

      // For very short context-dependent responses ("yes", "no", "ok") prepend
      // the last assistant message so the model has context
      const history = options.conversationHistory || [];
      const isShort = messageToClassify.length < 15 &&
        /^(yes|no|ok|sure|yeah|nope|yep|nah|maybe|definitely|absolutely|correct|right|wrong)$/i
          .test(messageToClassify);
      if (isShort && history.length > 0) {
        const lastAssistant = history.slice().reverse().find(m => m.role === 'assistant');
        if (lastAssistant) {
          messageToClassify = `[Context: ${lastAssistant.content.slice(0, 100)}] ${messageToClassify}`;
        }
      }

      // ── Run the classifier ────────────────────────────────────────────────
      const rawScores = await this.classifier(messageToClassify);

      // rawScores is an array like [{label: 'LABEL_0', score: 0.97}, ...]
      // Map back to intent names
      const scores = {};
      for (const { label, score } of rawScores) {
        // Handle both 'LABEL_N' format (default HuggingFace) and direct label names
        const idx = label.startsWith('LABEL_') ? parseInt(label.slice(6), 10) : -1;
        const intentName = idx >= 0 ? INTENT_LABELS[idx] : label;
        if (intentName) scores[intentName] = score;
      }

      // Remove screen_intelligence when highlighted text is present (upstream override)
      if (hasHighlight || options.excludeScreenIntelligence) {
        delete scores.screen_intelligence;
      }

      // ── Entity extraction (kept for context / suggested responses) ────────
      const entities = options.includeEntities !== false
        ? await this.extractEntities(message)
        : [];

      // ── Apply light entity boosting (small corrections only) ──────────────
      this._applyEntityBoosts(scores, entities, message);

      // ── Pick top intent ───────────────────────────────────────────────────
      const intent = Object.entries(scores).sort((a, b) => b[1] - a[1])[0][0];
      const confidence = scores[intent];

      const suggestedResponse = options.includeSuggestedResponse !== false
        ? IntentResponses.getSuggestedResponse(intent, message, entities)
        : null;

      return {
        intent,
        confidence,
        entities,
        suggestedResponse,
        parser: 'distilbert-finetuned',
        metadata: {
          processingTimeMs: Date.now() - startTime,
          modelVersion: 'distilbert-base-uncased-thinkdrop-v1',
          scores,
        },
      };
    } catch (error) {
      console.error('DistilBERT fine-tuned parsing failed:', error);
      throw error;
    }
  }

  /**
   * Light entity boosting — only for cases the model can't learn from text alone
   * (e.g. presence of a URL always means browser automation).
   * Keep this minimal — the fine-tuned model handles intent from context.
   */
  _applyEntityBoosts(scores, entities, message) {
    const lower = message.toLowerCase();

    // URL → command_automate
    if (/https?:\/\/|www\.\S+\.(com|org|io|ai|net|co|dev)\b/i.test(message)) {
      scores.command_automate = Math.max(scores.command_automate || 0, 0.85);
    }

    // Explicit screenshot/screen-read language → screen_intelligence
    // Boost to 0.95 to beat any command_automate score the model may produce.
    if (/\b(what'?s on my screen|read the screen|screenshot|screen capture|what does (it|the screen) say)\b/i.test(lower)) {
      scores.screen_intelligence = Math.max(scores.screen_intelligence || 0, 0.95);
    }

    // Single-word or 2-word message that matches a known app name → command_automate (shell.run open -a)
    // app_control_start is ONLY for explicit "take control / control mode" activation phrases
    const words = message.trim().split(/\s+/);
    if (words.length <= 3 && /^(obsidian|linear|figma|warp|notion|slack|zoom|spotify|discord|vscode|vs code|safari|firefox|chrome|calendar|mail|messages|facetime|finder|terminal|xcode|pycharm|intellij|datagrip|tableau|arc|brave|cursor|replit|codepen|postman|insomnia|raycast|alfred|things|todoist|fantastical|bear|craft|capacities|logseq|roam|anki|day one|dayone)$/i.test(message.trim())) {
      scores.command_automate = Math.max(scores.command_automate || 0, 0.85);
    }
  }

  // ── Entity extraction (unchanged from original — Compromise NLP) ────────────
  async extractEntities(message) {
    try {
      const doc = nlp(message);
      const entities = [];

      doc.people().forEach(p => {
        entities.push({ text: p.text(), type: 'person', start: 0, end: 0 });
      });
      doc.places().forEach(p => {
        entities.push({ text: p.text(), type: 'location', start: 0, end: 0 });
      });
      doc.organizations().forEach(o => {
        entities.push({ text: o.text(), type: 'organization', start: 0, end: 0 });
      });

      // Temporal entities
      const temporal = this._extractTemporalEntities(message);
      entities.push(...temporal);

      return entities;
    } catch (err) {
      return [];
    }
  }

  _extractTemporalEntities(message) {
    const entities = [];
    const patterns = [
      { re: /\b(today|tomorrow|yesterday)\b/gi, type: 'date' },
      { re: /\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b/gi, type: 'date' },
      { re: /\b(january|february|march|april|may|june|july|august|september|october|november|december)\b/gi, type: 'date' },
      { re: /\b\d{1,2}(:\d{2})?\s*(am|pm)\b/gi, type: 'time' },
      { re: /\bin\s+\d+\s+(minutes?|hours?|days?|weeks?)\b/gi, type: 'duration' },
    ];
    for (const { re, type } of patterns) {
      let m;
      while ((m = re.exec(message)) !== null) {
        entities.push({ text: m[0], type, start: m.index, end: m.index + m[0].length });
      }
    }
    return entities;
  }
}

module.exports = DistilBertIntentParser;
