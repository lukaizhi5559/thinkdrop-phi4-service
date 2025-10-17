/**
 * Embedding Generation Service
 * Generates text embeddings for semantic search
 */

const { pipeline } = require('@xenova/transformers');

class EmbeddingService {
  constructor() {
    this.embedder = null;
    this.initialized = false;
  }

  async initialize() {
    if (this.initialized) return;
    
    console.log('🚀 Initializing Embedding Service...');
    
    const embeddingModel = process.env.EMBEDDING_MODEL || 'Xenova/all-MiniLM-L6-v2';
    this.embedder = await pipeline('feature-extraction', embeddingModel);
    
    this.initialized = true;
    console.log('✅ Embedding Service initialized');
  }

  async generateEmbedding(text, options = {}) {
    if (!this.initialized) {
      await this.initialize();
    }

    try {
      const output = await this.embedder(text, {
        pooling: options.pooling || 'mean',
        normalize: options.normalize !== false
      });
      
      const embedding = Array.from(output.data);
      
      return {
        embedding,
        dimensions: embedding.length,
        model: process.env.EMBEDDING_MODEL || 'Xenova/all-MiniLM-L6-v2'
      };
    } catch (error) {
      console.error('Embedding generation failed:', error);
      throw new Error(`Embedding generation failed: ${error.message}`);
    }
  }
}

module.exports = new EmbeddingService();
