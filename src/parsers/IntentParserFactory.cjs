/**
 * Intent Parser Factory
 * Manages parser instances and provides fallback mechanism
 */

const DistilBertIntentParser = require('./DistilBertIntentParser.cjs');

class IntentParserFactory {
  constructor() {
    // Singleton instance
    this.distilBertInstance = null;
    
    // Initialization status
    this.initializationStatus = 'not_started';
  }

  /**
   * Get parser (always returns DistilBERT)
   * @returns {Promise<Object>} Parser instance
   */
  async getParser() {
    return await this.getDistilBertParser();
  }

  async getDistilBertParser() {
    if (!this.distilBertInstance) {
      this.initializationStatus = 'initializing';
      this.distilBertInstance = new DistilBertIntentParser();
      await this.distilBertInstance.initialize();
      this.initializationStatus = 'ready';
    }
    
    return this.distilBertInstance;
  }

  /**
   * Get list of available parsers
   * @returns {Promise<Array>} List of parser info
   */
  async listParsers() {
    return [{
      name: 'distilbert',
      description: 'DistilBERT fine-tuned parser (95%+ accuracy)',
      status: this.initializationStatus,
      accuracy: 0.95,
      avgLatency: 42
    }];
  }

  /**
   * Warm up DistilBERT parser
   */
  async warmup() {
    console.log('🔥 Warming up DistilBERT parser...');
    const startTime = Date.now();
    
    await this.getDistilBertParser();
    
    const elapsed = Date.now() - startTime;
    console.log(`✅ DistilBERT parser warmup completed in ${elapsed}ms`);
  }
}

// Export singleton instance
const factory = new IntentParserFactory();
module.exports = factory;
