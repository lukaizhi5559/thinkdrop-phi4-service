/**
 * Health Check and Capabilities Routes
 */

const express = require('express');
const router = express.Router();
const { metricsCollector } = require('../middleware/metrics');
const intentParsingService = require('../services/intentParsing');

router.get('/service.health', async (req, res) => {
  try {
    const parsers = await intentParsingService.listParsers();
    const metrics = metricsCollector.getMetrics();
    
    const parserStatus = {};
    parsers.forEach(p => {
      parserStatus[p.name] = p.status;
    });
    
    res.json({
      service: 'phi4',
      version: '1.0.0',
      status: 'up',
      uptime: metrics.uptime,
      parsers: parserStatus,
      models: {
        embedding: 'loaded',
        ner: 'loaded',
        phi4: 'placeholder'
      },
      metrics: {
        totalRequests: metrics.totalRequests,
        errorRate: metrics.errorRate,
        avgResponseTime: metrics.avgResponseTime,
        parserUsage: metrics.requestsByParser
      }
    });
  } catch (error) {
    res.status(503).json({
      service: 'phi4',
      version: '1.0.0',
      status: 'degraded',
      error: error.message
    });
  }
});

router.get('/service.capabilities', (req, res) => {
  res.json({
    service: 'phi4',
    version: '1.0.0',
    capabilities: {
      actions: [
        {
          name: 'intent.parse',
          description: 'Parse user message and classify intent',
          inputSchema: {
            message: 'string (required)',
            context: 'object (optional)',
            options: 'object (optional)'
          }
        },
        {
          name: 'general.answer',
          description: 'Generate answer for general knowledge questions',
          inputSchema: {
            query: 'string (required)',
            context: 'object (optional)',
            options: 'object (optional)'
          }
        },
        {
          name: 'entity.extract',
          description: 'Extract entities from text using NER',
          inputSchema: {
            text: 'string (required)',
            entityTypes: 'array (optional)',
            options: 'object (optional)'
          }
        },
        {
          name: 'embedding.generate',
          description: 'Generate text embeddings',
          inputSchema: {
            text: 'string (required)',
            model: 'string (optional)',
            options: 'object (optional)'
          }
        },
        {
          name: 'parser.list',
          description: 'List available parsers',
          inputSchema: {}
        }
      ],
      features: [
        'intent-classification',
        'entity-extraction',
        'general-qa',
        'embeddings',
        'multi-parser',
        'confidence-scoring',
        'suggested-responses'
      ],
      parsers: ['distilbert', 'hybrid', 'fast', 'original'],
      models: {
        embedding: process.env.EMBEDDING_MODEL || 'Xenova/all-MiniLM-L6-v2',
        ner: process.env.NER_MODEL || 'Xenova/bert-base-multilingual-cased-ner-hrl',
        llm: 'phi-4'
      }
    }
  });
});

module.exports = router;
