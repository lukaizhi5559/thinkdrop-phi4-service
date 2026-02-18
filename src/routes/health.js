/**
 * Health Check and Capabilities Routes
 */

const express = require('express');
const router = express.Router();
const intentParsingService = require('../services/intentParsing');

const startTime = Date.now();

router.get('/service.health', async (req, res) => {
  try {
    const parsers = await intentParsingService.listParsers();
    const uptime = Math.floor((Date.now() - startTime) / 1000);
    
    const parserStatus = {};
    parsers.forEach(p => {
      parserStatus[p.name] = p.status;
    });
    
    res.json({
      service: 'intent-parsing',
      version: '1.0.0',
      status: 'up',
      uptime,
      parsers: parserStatus
    });
  } catch (error) {
    res.status(503).json({
      service: 'intent-parsing',
      version: '1.0.0',
      status: 'degraded',
      error: error.message
    });
  }
});

router.get('/service.capabilities', (req, res) => {
  res.json({
    service: 'intent-parsing',
    version: '1.0.0',
    capabilities: {
      actions: [
        {
          name: 'intent.parse',
          description: 'Parse user message and classify intent using DistilBERT',
          inputSchema: {
            message: 'string (required)',
            context: 'object (optional)',
            options: 'object (optional)'
          }
        }
      ],
      features: [
        'intent-classification',
        'confidence-scoring',
        'suggested-responses',
        'entity-extraction'
      ],
      parsers: ['distilbert'],
      models: {
        distilbert: 'Xenova/distilbert-base-uncased'
      }
    }
  });
});

module.exports = router;
