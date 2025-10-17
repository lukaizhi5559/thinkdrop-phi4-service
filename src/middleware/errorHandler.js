/**
 * Error Handler Middleware
 * Catches and formats errors
 */

const errorHandler = (err, req, res, next) => {
  console.error('Error:', err);
  
  // Determine error code and message
  let errorCode = 'INTERNAL_ERROR';
  let statusCode = 500;
  let retryable = false;
  
  if (err.name === 'ValidationError') {
    errorCode = 'INVALID_REQUEST';
    statusCode = 400;
  } else if (err.message.includes('not found')) {
    errorCode = 'NOT_FOUND';
    statusCode = 404;
  } else if (err.message.includes('timeout')) {
    errorCode = 'TIMEOUT';
    statusCode = 504;
    retryable = true;
  } else if (err.message.includes('parser') || err.message.includes('model')) {
    errorCode = 'SERVICE_UNAVAILABLE';
    statusCode = 503;
    retryable = true;
  }
  
  res.status(statusCode).json({
    version: 'mcp.v1',
    status: 'error',
    error: {
      code: errorCode,
      message: err.message || 'An unexpected error occurred',
      retryable,
      stack: process.env.NODE_ENV === 'development' ? err.stack : undefined
    }
  });
};

module.exports = { errorHandler };
