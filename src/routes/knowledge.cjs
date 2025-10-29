/**
 * Knowledge-Based Answer Routes
 * Endpoint for answering questions using Phi4's built-in knowledge
 */

const express = require('express');
const router = express.Router();
const KnowledgeService = require('../services/knowledge.cjs');
const { validateKnowledgeRequest } = require('../middleware/validation.js');

const knowledgeService = new KnowledgeService();

/**
 * POST /knowledge.answer
 * Answer a factual question using built-in knowledge
 */
router.post('/knowledge.answer', validateKnowledgeRequest, async (req, res, next) => {
  try {
    const { query, options } = req.body.payload;
    
    console.log('Knowledge answer request:', { query, options });
    
    // Validate query exists
    if (!query) {
      throw new Error('query is required in payload');
    }
    
    // Answer from knowledge
    const result = await knowledgeService.answerFromKnowledge(query, options || {});
    
    res.json({
      version: 'mcp.v1',
      service: 'phi4',
      action: 'knowledge.answer',
      requestId: req.body.requestId,
      status: 'ok',
      data: result,
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
