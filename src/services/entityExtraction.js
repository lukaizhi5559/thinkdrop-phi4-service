/**
 * Entity Extraction Service
 * Handles NER-based entity extraction
 */

const { pipeline } = require('@xenova/transformers');

class EntityExtractionService {
  constructor() {
    this.nerClassifier = null;
    this.initialized = false;
  }

  async initialize() {
    if (this.initialized) return;
    
    console.log('🚀 Initializing Entity Extraction Service...');
    
    const nerModel = process.env.NER_MODEL || 'Xenova/bert-base-multilingual-cased-ner-hrl';
    this.nerClassifier = await pipeline('token-classification', nerModel, {
      grouped_entities: true
    });
    
    this.initialized = true;
    console.log('✅ Entity Extraction Service initialized');
  }

  async extractEntities(text, options = {}) {
    if (!this.initialized) {
      await this.initialize();
    }

    try {
      const nerResults = await this.nerClassifier(text);
      
      const entities = nerResults.map(entity => ({
        type: this.mapEntityType(entity.entity_group),
        value: entity.word,
        entity_type: entity.entity_group,
        confidence: entity.score,
        start: entity.start,
        end: entity.end
      }));
      
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

  mapEntityType(nerType) {
    const mapping = {
      'PER': 'person',
      'PERSON': 'person',
      'LOC': 'location',
      'GPE': 'location',
      'ORG': 'organization',
      'DATE': 'datetime',
      'TIME': 'datetime',
      'MISC': 'other'
    };
    
    return mapping[nerType] || nerType.toLowerCase();
  }
}

module.exports = new EntityExtractionService();
