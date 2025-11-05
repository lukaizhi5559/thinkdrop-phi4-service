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
    
    console.log('Intent parse request:', { message, hasContext: !!context, hasConversationHistory: !!(context?.conversationHistory), options });
    
    // Validate message exists
    if (!message) {
      throw new Error('message is required in payload');
    }
    
    // Merge context into options for the parser
    const parsingOptions = {
      ...(options || {}),
      conversationHistory: context?.conversationHistory || []
    };
    
    // Parse intent
    const result = await intentParsingService.parseIntent(message, parsingOptions);
    
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
