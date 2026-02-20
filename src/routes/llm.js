/**
 * LLM Generation Routes for Phi4 Service
 *
 * Adds the missing endpoints called by answer.cjs and storeConversation.cjs:
 *   POST /general.answer          - blocking answer generation (Ollama, with fallback)
 *   POST /general.answer.stream   - SSE streaming answer generation (Ollama, with fallback)
 *   POST /entity.extract          - named entity extraction (Ollama → compromise NLP fallback)
 *   POST /embedding.generate      - text embeddings via @xenova/transformers (no Ollama needed)
 *
 * Backend: Ollama (same as command-service) with graceful degradation when not running.
 * Model is auto-selected from OLLAMA_MODEL env var (default: qwen2.5:3b).
 */

const express = require('express');
const router = express.Router();

// Lazy-load Ollama to avoid startup failure if not installed
let _ollama = null;
function getOllama() {
  if (!_ollama) {
    const { Ollama } = require('ollama');
    _ollama = new Ollama({ host: process.env.OLLAMA_HOST || 'http://localhost:11434' });
  }
  return _ollama;
}

const OLLAMA_MODEL = process.env.OLLAMA_MODEL || 'qwen2.5:3b';

// Cached embedding pipeline (reuses the same model already loaded by DistilBERT parser)
let _embeddingPipeline = null;
async function getEmbeddingPipeline() {
  if (!_embeddingPipeline) {
    const { pipeline } = require('@xenova/transformers');
    _embeddingPipeline = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  }
  return _embeddingPipeline;
}

/**
 * Check if Ollama server is reachable (cached for 30s to avoid repeated probes)
 */
let _ollamaAvailable = null;
let _ollamaCheckedAt = 0;
async function isOllamaAvailable() {
  const now = Date.now();
  if (_ollamaAvailable !== null && now - _ollamaCheckedAt < 30000) {
    return _ollamaAvailable;
  }
  try {
    const net = require('net');
    const host = process.env.OLLAMA_HOST || 'http://localhost:11434';
    const url = new URL(host);
    const port = parseInt(url.port) || 11434;
    const hostname = url.hostname || '127.0.0.1';
    await new Promise((resolve, reject) => {
      const socket = net.createConnection({ host: hostname, port });
      const timer = setTimeout(() => {
        socket.destroy();
        reject(new Error('connect timeout'));
      }, 1500);
      socket.on('connect', () => {
        clearTimeout(timer);
        socket.destroy();
        resolve();
      });
      socket.on('error', (err) => {
        clearTimeout(timer);
        reject(err);
      });
    });
    _ollamaAvailable = true;
  } catch {
    _ollamaAvailable = false;
  }
  _ollamaCheckedAt = now;
  return _ollamaAvailable;
}

/**
 * Build a system prompt for general Q&A
 */
function buildSystemPrompt(systemInstruction) {
  return systemInstruction ||
    'You are ThinkDrop AI, a helpful assistant. Answer clearly and concisely.';
}

/**
 * Fallback entity extraction using compromise NLP (no Ollama required)
 */
function extractEntitiesWithNLP(text) {
  const nlp = require('compromise');
  const doc = nlp(text.substring(0, 2000));
  const entities = [];

  doc.people().forEach(p => {
    const t = p.text().trim();
    if (t) entities.push({ text: t, type: 'PERSON', confidence: 0.75 });
  });
  doc.organizations().forEach(o => {
    const t = o.text().trim();
    if (t) entities.push({ text: t, type: 'ORG', confidence: 0.70 });
  });
  doc.places().forEach(p => {
    const t = p.text().trim();
    if (t) entities.push({ text: t, type: 'LOCATION', confidence: 0.70 });
  });
  doc.dates().forEach(d => {
    const t = d.text().trim();
    if (t) entities.push({ text: t, type: 'DATE', confidence: 0.80 });
  });

  // Deduplicate by text
  const seen = new Set();
  return entities.filter(e => {
    if (seen.has(e.text)) return false;
    seen.add(e.text);
    return true;
  });
}

/**
 * POST /general.answer
 * Blocking answer generation
 *
 * Payload: { query, systemInstruction?, conversationHistory?, context? }
 */
router.post('/general.answer', async (req, res) => {
  const startTime = Date.now();
  const { requestId, payload } = req.body;
  const {
    query,
    systemInstruction,
    conversationHistory = []
  } = payload || {};

  if (!query) {
    return res.status(400).json({
      version: 'mcp.v1',
      service: 'phi4',
      action: 'general.answer',
      requestId,
      status: 'error',
      error: { code: 'INVALID_REQUEST', message: 'payload.query is required', retryable: false }
    });
  }

  try {
    const ollamaUp = await isOllamaAvailable();
    if (!ollamaUp) {
      const elapsedMs = Date.now() - startTime;
      const answer = `I'm currently running in offline mode (Ollama LLM service is not running). To enable full AI responses, please start Ollama: \`ollama serve\` and ensure the model is pulled: \`ollama pull ${OLLAMA_MODEL}\`.`;
      return res.json({
        version: 'mcp.v1',
        service: 'phi4',
        action: 'general.answer',
        requestId,
        status: 'ok',
        data: { answer, model: 'offline', elapsedMs },
        error: null,
        metrics: { elapsedMs }
      });
    }

    const ollama = getOllama();

    // Build messages array
    const messages = [
      { role: 'system', content: buildSystemPrompt(systemInstruction) }
    ];

    // Inject recent conversation history
    for (const turn of conversationHistory.slice(-10)) {
      if (turn.role && turn.content) {
        messages.push({ role: turn.role === 'ai' ? 'assistant' : turn.role, content: turn.content });
      }
    }

    messages.push({ role: 'user', content: query });

    const response = await ollama.chat({
      model: OLLAMA_MODEL,
      messages,
      stream: false,
      options: { temperature: 0.7, num_predict: 2048 }
    });

    const answer = response.message?.content || '';
    const elapsedMs = Date.now() - startTime;

    res.json({
      version: 'mcp.v1',
      service: 'phi4',
      action: 'general.answer',
      requestId,
      status: 'ok',
      data: { answer, model: OLLAMA_MODEL, elapsedMs },
      error: null,
      metrics: { elapsedMs }
    });
  } catch (err) {
    console.error('[phi4/general.answer] Error:', err.message);
    res.status(500).json({
      version: 'mcp.v1',
      service: 'phi4',
      action: 'general.answer',
      requestId,
      status: 'error',
      error: { code: 'LLM_ERROR', message: err.message, retryable: true }
    });
  }
});

/**
 * POST /general.answer.stream
 * SSE streaming answer generation
 *
 * Payload: { query, systemInstruction?, conversationHistory?, context? }
 * Response: text/event-stream with data: {"token":"..."} lines, ending with data: [DONE]
 */
router.post('/general.answer.stream', async (req, res) => {
  const { requestId, payload } = req.body;
  const {
    query,
    systemInstruction,
    conversationHistory = []
  } = payload || {};

  if (!query) {
    return res.status(400).json({
      version: 'mcp.v1',
      service: 'phi4',
      action: 'general.answer.stream',
      requestId,
      status: 'error',
      error: { code: 'INVALID_REQUEST', message: 'payload.query is required', retryable: false }
    });
  }

  // Set SSE headers
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();

  try {
    const ollamaUp = await isOllamaAvailable();
    if (!ollamaUp) {
      const fallback = `I'm currently running in offline mode (Ollama LLM service is not running). Start Ollama with \`ollama serve\` to enable full AI responses.`;
      res.write(`data: ${JSON.stringify({ type: 'token', token: fallback })}\n\n`);
      res.write(`data: ${JSON.stringify({ type: 'done', metrics: { elapsedMs: 0 } })}\n\n`);
      return res.end();
    }

    const ollama = getOllama();

    const messages = [
      { role: 'system', content: buildSystemPrompt(systemInstruction) }
    ];

    for (const turn of conversationHistory.slice(-10)) {
      if (turn.role && turn.content) {
        messages.push({ role: turn.role === 'ai' ? 'assistant' : turn.role, content: turn.content });
      }
    }

    messages.push({ role: 'user', content: query });

    const stream = await ollama.chat({
      model: OLLAMA_MODEL,
      messages,
      stream: true,
      options: { temperature: 0.7, num_predict: 2048 }
    });

    let fullAnswer = '';
    const streamStart = Date.now();
    res.write(`data: ${JSON.stringify({ type: 'start', timestamp: streamStart })}\n\n`);

    for await (const chunk of stream) {
      const token = chunk.message?.content || '';
      if (token) {
        fullAnswer += token;
        res.write(`data: ${JSON.stringify({ type: 'token', token })}\n\n`);
      }
    }

    res.write(`data: ${JSON.stringify({ type: 'done', metrics: { elapsedMs: Date.now() - streamStart, tokenCount: fullAnswer.length } })}\n\n`);
    res.end();
  } catch (err) {
    console.error('[phi4/general.answer.stream] Error:', err.message);
    res.write(`data: ${JSON.stringify({ type: 'error', error: err.message })}\n\n`);
    res.write(`data: ${JSON.stringify({ type: 'done', metrics: { elapsedMs: 0 } })}\n\n`);
    res.end();
  }
});

/**
 * POST /entity.extract
 * Extract named entities from text
 *
 * Payload: { text }
 * Response: { entities: [{ text, type, confidence }] }
 */
router.post('/entity.extract', async (req, res) => {
  const startTime = Date.now();
  const { requestId, payload } = req.body;
  const { text } = payload || {};

  if (!text) {
    return res.status(400).json({
      version: 'mcp.v1',
      service: 'phi4',
      action: 'entity.extract',
      requestId,
      status: 'error',
      error: { code: 'INVALID_REQUEST', message: 'payload.text is required', retryable: false }
    });
  }

  try {
    const ollamaUp = await isOllamaAvailable();

    let entities = [];
    let model = 'compromise-nlp';

    if (ollamaUp) {
      try {
        const ollama = getOllama();
        const prompt = `Extract named entities from the following text. Return ONLY a JSON array of objects with fields: text, type (PERSON/ORG/LOCATION/DATE/PRODUCT/CONCEPT/OTHER), confidence (0.0-1.0). No explanation, just the JSON array.\n\nText: ${text.substring(0, 2000)}\n\nJSON array:`;
        const response = await ollama.chat({
          model: OLLAMA_MODEL,
          messages: [{ role: 'user', content: prompt }],
          stream: false,
          options: { temperature: 0.1, num_predict: 512 }
        });
        const content = response.message?.content || '[]';
        const match = content.match(/\[[\s\S]*\]/);
        if (match) entities = JSON.parse(match[0]);
        model = OLLAMA_MODEL;
      } catch (ollamaErr) {
        console.warn('[phi4/entity.extract] Ollama failed, using NLP fallback:', ollamaErr.message);
        entities = extractEntitiesWithNLP(text);
      }
    } else {
      console.warn('[phi4/entity.extract] Ollama not available, using NLP fallback');
      entities = extractEntitiesWithNLP(text);
    }

    const elapsedMs = Date.now() - startTime;

    res.json({
      version: 'mcp.v1',
      service: 'phi4',
      action: 'entity.extract',
      requestId,
      status: 'ok',
      data: { entities, model, elapsedMs },
      error: null,
      metrics: { elapsedMs }
    });
  } catch (err) {
    console.error('[phi4/entity.extract] Error:', err.message);
    res.status(500).json({
      version: 'mcp.v1',
      service: 'phi4',
      action: 'entity.extract',
      requestId,
      status: 'error',
      error: { code: 'LLM_ERROR', message: err.message, retryable: true }
    });
  }
});

/**
 * POST /embedding.generate
 * Generate text embeddings using @xenova/transformers (no Ollama required)
 *
 * Payload: { text, options?: { normalize?, pooling? } }
 * Response: { embedding: number[], dimensions: number }
 */
router.post('/embedding.generate', async (req, res) => {
  const startTime = Date.now();
  const { requestId, payload } = req.body;
  const { text, options = {} } = payload || {};

  if (!text) {
    return res.status(400).json({
      version: 'mcp.v1',
      service: 'phi4',
      action: 'embedding.generate',
      requestId,
      status: 'error',
      error: { code: 'INVALID_REQUEST', message: 'payload.text is required', retryable: false }
    });
  }

  try {
    const embedder = await getEmbeddingPipeline();
    const output = await embedder(text.substring(0, 8192), {
      pooling: options.pooling || 'mean',
      normalize: options.normalize !== false
    });

    // Convert typed array to plain JS array
    const embedding = Array.from(output.data);
    const elapsedMs = Date.now() - startTime;

    res.json({
      version: 'mcp.v1',
      service: 'phi4',
      action: 'embedding.generate',
      requestId,
      status: 'ok',
      data: { embedding, dimensions: embedding.length, model: 'all-MiniLM-L6-v2', elapsedMs },
      error: null,
      metrics: { elapsedMs }
    });
  } catch (err) {
    console.error('[phi4/embedding.generate] Error:', err.message);
    res.status(500).json({
      version: 'mcp.v1',
      service: 'phi4',
      action: 'embedding.generate',
      requestId,
      status: 'error',
      error: { code: 'EMBEDDING_ERROR', message: err.message, retryable: true }
    });
  }
});

module.exports = router;
