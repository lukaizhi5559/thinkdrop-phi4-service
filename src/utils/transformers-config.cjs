/**
 * Configuration for @xenova/transformers library
 * Sets up WASM backend and model caching
 */

const { env } = require('@xenova/transformers');
const path = require('path');

// Configure model cache directory
const MODEL_CACHE_DIR = process.env.MODEL_CACHE_DIR || './models';
env.cacheDir = path.resolve(MODEL_CACHE_DIR);

// Use local models when available
env.allowLocalModels = true;

// Disable remote model fetching in production (optional)
if (process.env.NODE_ENV === 'production') {
  env.allowRemoteModels = true; // Still allow for first-time download
}

// Configure WASM backend
env.backends.onnx.wasm.numThreads = 4;

console.log(`📦 Transformers cache directory: ${env.cacheDir}`);

module.exports = { env };
