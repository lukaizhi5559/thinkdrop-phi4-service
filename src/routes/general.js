/**
 * General Q&A Routes
 */

const express = require('express');
const router = express.Router();
const llmService = require('../services/llm');
const { validateGeneralAnswerRequest } = require('../middleware/validation');

router.post('/general.answer', validateGeneralAnswerRequest, async (req, res, next) => {
  try {
    const { query, context, options } = req.body.payload;
    
    // Generate answer
    const result = await llmService.generateAnswer(query, options || {});
    
    res.json({
      version: 'mcp.v1',
      service: 'phi4',
      action: 'general.answer',
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
