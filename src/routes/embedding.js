/**
 * Embedding Routes for Phi4 Service
 *
 * POST /embedding.generate — text embeddings via @xenova/transformers
 * (all-MiniLM-L6-v2). No Ollama required. Used by conversation-service
 * semantic search and stategraph memory ranking.
 */

const express = require('express');
const router = express.Router();

// Cached embedding pipeline (lazy singleton)
let _embeddingPipeline = null;
async function getEmbeddingPipeline() {
  if (!_embeddingPipeline) {
    const { pipeline } = require('@xenova/transformers');
    _embeddingPipeline = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  }
  return _embeddingPipeline;
}

/**
 * POST /embedding.generate
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
