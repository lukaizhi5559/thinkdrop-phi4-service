/**
 * Phi4 MCP Microservice
 * Main server file
 */

require('dotenv').config();
require('./utils/transformers-config.cjs');

const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');

// Middleware
const { validateApiKey } = require('./middleware/auth');
const { validateMCPRequest } = require('./middleware/validation');
const { metricsMiddleware } = require('./middleware/metrics');
const { errorHandler } = require('./middleware/errorHandler');

// Routes
const healthRoutes = require('./routes/health');
const intentRoutes = require('./routes/intent');
const generalRoutes = require('./routes/general');
const entityRoutes = require('./routes/entity');
const embeddingRoutes = require('./routes/embedding');
const parserRoutes = require('./routes/parser');
const knowledgeRoutes = require('./routes/knowledge.cjs');

// Services
const intentParsingService = require('./services/intentParsing');

const app = express();
const PORT = process.env.PORT || 3003;
const HOST = process.env.HOST || '0.0.0.0';

// Security middleware
app.use(helmet());

// CORS configuration
const allowedOrigins = process.env.ALLOWED_ORIGINS
  ? process.env.ALLOWED_ORIGINS.split(',')
  : ['http://localhost:3000', 'http://localhost:5173'];

app.use(cors({
  origin: (origin, callback) => {
    // Allow requests with no origin (like mobile apps or curl)
    if (!origin) return callback(null, true);
    
    if (allowedOrigins.includes(origin)) {
      callback(null, true);
    } else {
      callback(new Error('Not allowed by CORS'));
    }
  },
  credentials: true
}));

// Rate limiting
const limiter = rateLimit({
  windowMs: parseInt(process.env.RATE_LIMIT_WINDOW_MS) || 60000,
  max: parseInt(process.env.RATE_LIMIT_MAX_REQUESTS) || 100,
  message: {
    version: 'mcp.v1',
    status: 'error',
    error: {
      code: 'RATE_LIMIT_EXCEEDED',
      message: 'Too many requests, please try again later',
      retryable: true
    }
  }
});

app.use(limiter);

// Body parser
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Metrics middleware
app.use(metricsMiddleware);

// Health check routes (no auth required)
app.use('/', healthRoutes);

// API routes (require auth and MCP validation)
app.use('/', validateApiKey, validateMCPRequest, intentRoutes);
app.use('/', validateApiKey, validateMCPRequest, generalRoutes);
app.use('/', validateApiKey, validateMCPRequest, entityRoutes);
app.use('/', validateApiKey, validateMCPRequest, embeddingRoutes);
app.use('/', validateApiKey, validateMCPRequest, parserRoutes);
app.use('/', validateApiKey, validateMCPRequest, knowledgeRoutes);

// Error handler (must be last)
app.use(errorHandler);

// Initialize and start server
async function startServer() {
  try {
    console.log('🚀 Starting Phi4 MCP Service...');
    console.log(`   Environment: ${process.env.NODE_ENV || 'development'}`);
    console.log(`   Port: ${PORT}`);
    console.log(`   Host: ${HOST}`);
    
    // Warm up parsers if enabled
    if (process.env.MODEL_WARMUP_ON_START === 'true') {
      console.log('🔥 Warming up parsers...');
      await intentParsingService.warmup();
    }
    
    // Start server
    app.listen(PORT, HOST, () => {
      console.log('✅ Phi4 MCP Service is running');
      console.log(`   URL: http://${HOST}:${PORT}`);
      console.log(`   Health: http://${HOST}:${PORT}/service.health`);
      console.log(`   Capabilities: http://${HOST}:${PORT}/service.capabilities`);
      console.log('');
      console.log('📊 Available Actions:');
      console.log('   - POST /intent.parse');
      console.log('   - POST /general.answer');
      console.log('   - POST /knowledge.answer');
      console.log('   - POST /entity.extract');
      console.log('   - POST /embedding.generate');
      console.log('   - POST /parser.list');
    });
  } catch (error) {
    console.error('❌ Failed to start server:', error);
    process.exit(1);
  }
}

// Handle graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down gracefully...');
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('SIGINT received, shutting down gracefully...');
  process.exit(0);
});

// Start the server
startServer();

module.exports = app;
