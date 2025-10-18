/**
 * Intent Parsing Routes
 */

const express = require('express');
const router = express.Router();
const intentParsingService = require('../services/intentParsing');
const { validateIntentParseRequest } = require('../middleware/validation');

router.post('/intent.parse', validateIntentParseRequest, async (req, res, next) => {
  try {
    const { message, context, options } = req.body.payload;
    
    console.log('Intent parse request:', { message, options });
    
    // Validate message exists
    if (!message) {
      throw new Error('message is required in payload');
    }
    
    // Parse intent
    const result = await intentParsingService.parseIntent(message, options || {});
    
    res.json({
      version: 'mcp.v1',
      service: 'phi4',
      action: 'intent.parse',
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
