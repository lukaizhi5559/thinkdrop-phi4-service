/**
 * Request Validation Middleware
 * Validates MCP request structure and payload
 */

const validateMCPRequest = (req, res, next) => {
  const { body } = req;
  
  // Check MCP envelope structure
  if (!body.version || !body.service || !body.action) {
    return res.status(400).json({
      version: 'mcp.v1',
      status: 'error',
      error: {
        code: 'INVALID_REQUEST',
        message: 'Missing required MCP fields: version, service, action',
        retryable: false
      }
    });
  }
  
  // Validate version
  if (body.version !== 'mcp.v1') {
    return res.status(400).json({
      version: 'mcp.v1',
      status: 'error',
      error: {
        code: 'INVALID_VERSION',
        message: `Unsupported MCP version: ${body.version}`,
        retryable: false
      }
    });
  }
  
  // Validate service
  if (body.service !== 'phi4') {
    return res.status(400).json({
      version: 'mcp.v1',
      status: 'error',
      error: {
        code: 'INVALID_SERVICE',
        message: `Invalid service: ${body.service}. Expected: phi4`,
        retryable: false
      }
    });
  }
  
  // Validate payload exists
  if (!body.payload) {
    return res.status(400).json({
      version: 'mcp.v1',
      status: 'error',
      error: {
        code: 'INVALID_REQUEST',
        message: 'Missing payload',
        retryable: false
      }
    });
  }
  
  next();
};

const validateIntentParseRequest = (req, res, next) => {
  const { payload } = req.body;
  
  if (!payload.message) {
    return res.status(400).json({
      version: 'mcp.v1',
      status: 'error',
      error: {
        code: 'INVALID_REQUEST',
        message: 'Missing required field: message',
        retryable: false
      }
    });
  }
  
  if (typeof payload.message !== 'string') {
    return res.status(400).json({
      version: 'mcp.v1',
      status: 'error',
      error: {
        code: 'INVALID_REQUEST',
        message: 'Field "message" must be a string',
        retryable: false
      }
    });
  }
  
  if (payload.message.length > 50000) {
    return res.status(400).json({
      version: 'mcp.v1',
      status: 'error',
      error: {
        code: 'INVALID_REQUEST',
        message: 'Message exceeds maximum length of 50000 characters',
        retryable: false
      }
    });
  }
  
  next();
};

module.exports = {
  validateMCPRequest,
  validateIntentParseRequest
};
