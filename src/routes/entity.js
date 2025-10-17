/**
 * Entity Extraction Routes
 */

const express = require('express');
const router = express.Router();
const entityExtractionService = require('../services/entityExtraction');
const { validateEntityExtractRequest } = require('../middleware/validation');

router.post('/entity.extract', validateEntityExtractRequest, async (req, res, next) => {
  try {
    const { text, entityTypes, options } = req.body.payload;
    
    // Extract entities
    const entities = await entityExtractionService.extractEntities(text, {
      entityTypes,
      ...options
    });
    
    res.json({
      version: 'mcp.v1',
      service: 'phi4',
      action: 'entity.extract',
      requestId: req.body.requestId,
      status: 'ok',
      data: {
        entities,
        text,
        metadata: {
          model: process.env.NER_MODEL || 'Xenova/bert-base-multilingual-cased-ner-hrl',
          processingTimeMs: Date.now() - req.startTime,
          totalEntities: entities.length
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
