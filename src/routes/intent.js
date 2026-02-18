/**
 * Intent Parsing Routes
 */

const express = require('express');
const router = express.Router();
const intentParsingService = require('../services/intentParsing');
const { validateIntentParseRequest } = require('../middleware/validation');

router.post('/intent.parse', validateIntentParseRequest, async (req, res, next) => {
  const startTime = Date.now();
  
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
    
    const elapsedMs = Date.now() - startTime;
    console.log(`Intent parsed in ${elapsedMs}ms:`, result.intent);
    
    res.json({
      version: 'mcp.v1',
      service: 'phi4',
      action: 'intent.parse',
      requestId: req.body.requestId,
      status: 'ok',
      data: result,
      error: null,
      metrics: {
        elapsedMs
      }
    });
  } catch (error) {
    const elapsedMs = Date.now() - startTime;
    console.error(`Intent parse failed after ${elapsedMs}ms:`, error.message);
    next(error);
  }
});

module.exports = router;
