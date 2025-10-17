/**
 * Embedding Generation Routes
 */

const express = require('express');
const router = express.Router();
const embeddingService = require('../services/embeddings');
const { validateEmbeddingRequest } = require('../middleware/validation');

router.post('/embedding.generate', validateEmbeddingRequest, async (req, res, next) => {
  try {
    const { text, model, options } = req.body.payload;
    
    // Generate embedding
    const result = await embeddingService.generateEmbedding(text, options || {});
    
    res.json({
      version: 'mcp.v1',
      service: 'phi4',
      action: 'embedding.generate',
      requestId: req.body.requestId,
      status: 'ok',
      data: {
        ...result,
        metadata: {
          processingTimeMs: Date.now() - req.startTime,
          normalized: options?.normalize !== false,
          pooling: options?.pooling || 'mean'
        }
      },
      error: null,
      metrics: {
        elapsedMs: Date.now() - req.startTime
      }
    });
  } catch (error) {
    next(error);
  }
});

module.exports = router;
