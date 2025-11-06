/**
 * Entity Extraction Service
 * Uses DistilBERT parser's entity extraction for consistency
 */

const IntentParserFactory = require('../parsers/IntentParserFactory.cjs');

class EntityExtractionService {
  constructor() {
    this.parser = null;
    this.initialized = false;
  }

  async initialize() {
    if (this.initialized) return;
    
    console.log('🚀 Initializing Entity Extraction Service...');
    
    // Use the same parser as intent classification for consistency
    this.parser = await IntentParserFactory.getParser('distilbert');
    
    this.initialized = true;
    console.log('✅ Entity Extraction Service initialized');
  }

  async extractEntities(text, options = {}) {
    if (!this.initialized) {
      await this.initialize();
    }

    try {
      // Use the parser's extractEntities method (Compromise-based)
      const entities = await this.parser.extractEntities(text);
      
      // Filter by entity types if specified
      if (options.entityTypes && options.entityTypes.length > 0) {
        return entities.filter(e => options.entityTypes.includes(e.type));
      }
      
      return entities;
    } catch (error) {
      console.error('Entity extraction failed:', error);
      throw new Error(`Entity extraction failed: ${error.message}`);
    }
  }
}

module.exports = new EntityExtractionService();
