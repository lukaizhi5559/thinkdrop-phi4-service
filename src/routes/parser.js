/**
 * Parser Management Routes
 */

const express = require('express');
const router = express.Router();
const intentParsingService = require('../services/intentParsing');

router.post('/parser.list', async (req, res, next) => {
  try {
    const parsers = await intentParsingService.listParsers();
    
    res.json({
      version: 'mcp.v1',
      service: 'phi4',
      action: 'parser.list',
      requestId: req.body.requestId,
      status: 'ok',
      data: {
        parsers,
        default: process.env.DEFAULT_PARSER || 'distilbert'
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
