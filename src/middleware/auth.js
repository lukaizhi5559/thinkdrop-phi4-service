/**
 * Authentication Middleware
 * Validates API key from Authorization header
 */

const validateApiKey = (req, res, next) => {
  const apiKey = process.env.API_KEY;
  
  // Skip auth in development if no API key is set
  if (!apiKey && process.env.NODE_ENV === 'development') {
    console.warn('⚠️ No API_KEY set, skipping authentication');
    return next();
  }
  
  const authHeader = req.headers.authorization;
  
  if (!authHeader) {
    return res.status(401).json({
      version: 'mcp.v1',
      status: 'error',
      error: {
        code: 'UNAUTHORIZED',
        message: 'Missing Authorization header',
        retryable: false
      }
    });
  }
  
  // Support both "Bearer TOKEN" and just "TOKEN"
  const token = authHeader.startsWith('Bearer ')
    ? authHeader.substring(7)
    : authHeader;
  
  if (token !== apiKey) {
    return res.status(401).json({
      version: 'mcp.v1',
      status: 'error',
      error: {
        code: 'UNAUTHORIZED',
        message: 'Invalid API key',
        retryable: false
      }
    });
  }
  
  next();
};

module.exports = { validateApiKey };
