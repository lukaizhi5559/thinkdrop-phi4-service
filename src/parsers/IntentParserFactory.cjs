/**
 * Intent Parser Factory
 * Manages parser instances and provides fallback mechanism
 */

const DistilBertIntentParser = require('./DistilBertIntentParser.cjs');
const FastIntentParser = require('./FastIntentParser.cjs');
const HybridIntentParser = require('./HybridIntentParser.cjs');
const IntentParser = require('./IntentParser.cjs');

class IntentParserFactory {
  constructor() {
    this.config = {
      useDistilBert: process.env.ENABLE_DISTILBERT !== 'false',
      useHybrid: process.env.ENABLE_HYBRID === 'true',
      useFast: process.env.ENABLE_FAST === 'true',
      useOriginal: process.env.ENABLE_ORIGINAL === 'true',
      defaultParser: process.env.DEFAULT_PARSER || 'distilbert'
    };
    
    // Singleton instances
    this.distilBertInstance = null;
    this.hybridInstance = null;
    this.fastInstance = null;
    this.originalInstance = null;
    
    // Initialization status
    this.initializationStatus = {
      distilbert: 'not_started',
      hybrid: 'not_started',
      fast: 'not_started',
      original: 'not_started'
    };
  }

  /**
   * Get parser by name
   * @param {string} parserName - Parser name (distilbert, fast, hybrid, original)
   * @returns {Promise<Object>} Parser instance
   */
  async getParser(parserName = null) {
    const name = parserName || this.config.defaultParser;
    
    try {
      switch (name.toLowerCase()) {
        case 'distilbert':
          return await this.getDistilBertParser();
        
        case 'fast':
          return await this.getFastParser();
        
        case 'hybrid':
          return await this.getHybridParser();
        
        case 'original':
          return await this.getOriginalParser();
        
        default:
          console.warn(`Unknown parser: ${name}, falling back to default`);
          return await this.getDefaultParser();
      }
    } catch (error) {
      console.error(`Failed to get ${name} parser:`, error.message);
      return await this.getFallbackParser();
    }
  }

  async getDistilBertParser() {
    if (!this.config.useDistilBert) {
      throw new Error('DistilBERT parser is disabled');
    }
    
    if (!this.distilBertInstance) {
      this.initializationStatus.distilbert = 'initializing';
      this.distilBertInstance = new DistilBertIntentParser();
      await this.distilBertInstance.initialize();
      this.initializationStatus.distilbert = 'ready';
    }
    
    return this.distilBertInstance;
  }

  async getFastParser() {
    if (!this.config.useFast) {
      throw new Error('Fast parser is disabled');
    }
    
    if (!this.fastInstance) {
      this.initializationStatus.fast = 'initializing';
      this.fastInstance = new FastIntentParser();
      await this.fastInstance.initialize();
      this.initializationStatus.fast = 'ready';
    }
    
    return this.fastInstance;
  }

  async getHybridParser() {
    if (!this.config.useHybrid) {
      throw new Error('Hybrid parser is disabled');
    }
    
    if (!this.hybridInstance) {
      this.initializationStatus.hybrid = 'initializing';
      this.hybridInstance = new HybridIntentParser();
      await this.hybridInstance.initialize();
      this.initializationStatus.hybrid = 'ready';
    }
    
    return this.hybridInstance;
  }

  async getOriginalParser() {
    if (!this.config.useOriginal) {
      throw new Error('Original parser is disabled');
    }
    
    if (!this.originalInstance) {
      this.initializationStatus.original = 'initializing';
      this.originalInstance = new IntentParser();
      await this.originalInstance.initialize();
      this.initializationStatus.original = 'ready';
    }
    
    return this.originalInstance;
  }

  async getDefaultParser() {
    // Try parsers in order of preference
    const fallbackOrder = ['distilbert', 'hybrid', 'fast', 'original'];
    
    for (const parserName of fallbackOrder) {
      try {
        return await this.getParser(parserName);
      } catch (error) {
        console.warn(`Failed to load ${parserName} parser, trying next...`);
      }
    }
    
    throw new Error('No parsers available');
  }

  async getFallbackParser() {
    // Fast parser is the most reliable fallback
    try {
      return await this.getFastParser();
    } catch (error) {
      console.error('All parsers failed, creating emergency Fast parser');
      const emergencyParser = new FastIntentParser();
      await emergencyParser.initialize();
      return emergencyParser;
    }
  }

  /**
   * Get list of available parsers
   * @returns {Promise<Array>} List of parser info
   */
  async listParsers() {
    const parsers = [];
    
    if (this.config.useDistilBert) {
      parsers.push({
        name: 'distilbert',
        description: 'DistilBERT fine-tuned parser (95%+ accuracy)',
        status: this.initializationStatus.distilbert,
        accuracy: 0.95,
        avgLatency: 42
      });
    }
    
    if (this.config.useHybrid) {
      parsers.push({
        name: 'hybrid',
        description: 'TensorFlow.js + USE + Natural + Compromise',
        status: this.initializationStatus.hybrid,
        accuracy: 0.88,
        avgLatency: 67
      });
    }
    
    if (this.config.useFast) {
      parsers.push({
        name: 'fast',
        description: 'Natural + Compromise (lightweight)',
        status: this.initializationStatus.fast,
        accuracy: 0.82,
        avgLatency: 15
      });
    }
    
    if (this.config.useOriginal) {
      parsers.push({
        name: 'original',
        description: 'Original heavy parser with embeddings',
        status: this.initializationStatus.original,
        accuracy: 0.85,
        avgLatency: 89
      });
    }
    
    return parsers;
  }

  /**
   * Warm up all enabled parsers
   */
  async warmup() {
    console.log('🔥 Warming up parsers...');
    const startTime = Date.now();
    
    const promises = [];
    
    if (this.config.useDistilBert) {
      promises.push(this.getDistilBertParser().catch(e => console.warn('DistilBERT warmup failed:', e.message)));
    }
    
    if (this.config.useFast) {
      promises.push(this.getFastParser().catch(e => console.warn('Fast warmup failed:', e.message)));
    }
    
    if (this.config.useHybrid) {
      promises.push(this.getHybridParser().catch(e => console.warn('Hybrid warmup failed:', e.message)));
    }
    
    if (this.config.useOriginal) {
      promises.push(this.getOriginalParser().catch(e => console.warn('Original warmup failed:', e.message)));
    }
    
    await Promise.allSettled(promises);
    
    const elapsed = Date.now() - startTime;
    console.log(`✅ Parser warmup completed in ${elapsed}ms`);
  }
}

// Export singleton instance
const factory = new IntentParserFactory();
module.exports = factory;
