/**
 * Intent Parsing Service
 * Handles intent classification using various parsers
 */

const IntentParserFactory = require('../parsers/IntentParserFactory.cjs');

class IntentParsingService {
  constructor() {
    this.factory = IntentParserFactory;
  }

  /**
   * Parse user message and classify intent
   * @param {string} message - User message
   * @param {Object} options - Parsing options
   * @returns {Promise<Object>} Parsed result
   */
  async parseIntent(message, options = {}) {
    console.log('parseIntent called with:', { 
      message: message, 
      messageType: typeof message,
      messageValue: JSON.stringify(message),
      options 
    });
    
    if (!message) {
      throw new Error('message parameter is required');
    }
    
    if (typeof message !== 'string') {
      throw new Error(`message must be a string, got ${typeof message}`);
    }
    
    const parserName = options.parser || process.env.DEFAULT_PARSER || 'distilbert';
    
    try {
      const parser = await this.factory.getParser(parserName);
      console.log('Parser obtained:', parserName);
      
      const result = await parser.parse(message, options);
      console.log('Parse result:', result);
      
      return result;
    } catch (error) {
      console.error('Intent parsing failed:', error);
      console.error('Error stack:', error.stack);
      throw new Error(`Intent parsing failed: ${error.message}`);
    }
  }

  /**
   * List available parsers
   * @returns {Promise<Array>} List of parsers
   */
  async listParsers() {
    return await this.factory.listParsers();
  }

  /**
   * Warm up parsers
   */
  async warmup() {
    if (process.env.MODEL_WARMUP_ON_START === 'true') {
      await this.factory.warmup();
    }
  }
}

module.exports = new IntentParsingService();
