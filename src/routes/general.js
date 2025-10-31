/**
 * General Q&A Routes
 */

const express = require('express');
const router = express.Router();
const llmService = require('../services/llm');
const { validateGeneralAnswerRequest } = require('../middleware/validation');

// Non-streaming endpoint (original)
router.post('/general.answer', validateGeneralAnswerRequest, async (req, res, next) => {
  try {
    const { query, context, options } = req.body.payload;
    
    // Generate answer with context (including memories)
    const result = await llmService.generateAnswer(query, { ...options, context });
    
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

// Streaming endpoint with Server-Sent Events (SSE)
router.post('/general.answer.stream', validateGeneralAnswerRequest, async (req, res) => {
  try {
    const { query, context, options } = req.body.payload;
    const requestId = req.body.requestId;
    
    // Set SSE headers
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('X-Accel-Buffering', 'no'); // Disable nginx buffering
    
    // Send initial metadata
    res.write(`data: ${JSON.stringify({
      type: 'start',
      requestId,
      timestamp: new Date().toISOString()
    })}\n\n`);
    
    let fullAnswer = '';
    let tokenCount = 0;
    
    // Stream answer token by token
    await llmService.generateAnswerStream(
      query,
      { ...options, context },
      (chunk) => {
        // Send each token as it arrives
        fullAnswer += chunk.token;
        tokenCount++;
        
        res.write(`data: ${JSON.stringify({
          type: 'token',
          token: chunk.token,
          tokenCount,
          timestamp: new Date().toISOString()
        })}\n\n`);
      }
    );
    
    // Send completion event
    res.write(`data: ${JSON.stringify({
      type: 'done',
      answer: fullAnswer,
      tokenCount,
      metrics: {
        elapsedMs: Date.now() - req.startTime
      },
      timestamp: new Date().toISOString()
    })}\n\n`);
    
    res.end();
  } catch (error) {
    // Send error event
    res.write(`data: ${JSON.stringify({
      type: 'error',
      error: error.message,
      timestamp: new Date().toISOString()
    })}\n\n`);
    res.end();
  }
});

module.exports = router;
