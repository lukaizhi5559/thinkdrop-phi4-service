/**
 * Original Intent Parser
 * Heavy parser with zero-shot classification and comprehensive NER
 * Accuracy: ~85%
 * Latency: ~89ms
 */

const { pipeline } = require('@xenova/transformers');
const MathUtils = require('../utils/MathUtils.cjs');

class NaturalLanguageIntentParser {
  constructor() {
    this.embedder = null;
    this.zeroShotClassifier = null;
    this.nerClassifier = null;
    this.initialized = false;
    this.seedEmbeddings = null;
    
    this.intentLabels = [
      'memory_store',
      'memory_retrieve',
      'command',
      'question',
      'greeting',
      'context'
    ];
    
    // Candidate labels for zero-shot classification
    this.candidateLabels = [
      'storing information',
      'retrieving information',
      'executing a command',
      'asking a question',
      'greeting',
      'asking about conversation history'
    ];
  }

  async initialize() {
    if (this.initialized) return;
    
    console.log('🚀 Initializing Original IntentParser...');
    const startTime = Date.now();
    
    try {
      // Load embedding model
      console.log('  Loading embedding model...');
      this.embedder = await pipeline(
        'feature-extraction',
        'Xenova/all-MiniLM-L6-v2'
      );
      
      // Load zero-shot classifier
      console.log('  Loading zero-shot classifier...');
      this.zeroShotClassifier = await pipeline(
        'zero-shot-classification',
        'Xenova/distilbert-base-uncased-mnli'
      );
      
      // Load NER classifier
      console.log('  Loading NER classifier...');
      this.nerClassifier = await pipeline(
        'token-classification',
        'Xenova/bert-base-multilingual-cased-ner-hrl',
        { grouped_entities: true }
      );
      
      // Pre-compute seed embeddings
      console.log('  Computing seed embeddings...');
      await this.initializeEmbeddings();
      
      this.initialized = true;
      const elapsed = Date.now() - startTime;
      console.log(`✅ Original IntentParser initialized in ${elapsed}ms`);
    } catch (error) {
      console.error('❌ Failed to initialize Original IntentParser:', error);
      throw error;
    }
  }

  async initializeEmbeddings() {
    const seedExamples = {
      memory_store: [
        "Remember I have a meeting tomorrow",
        "Save this information",
        "Don't forget my appointment"
      ],
      memory_retrieve: [
        "What meetings do I have?",
        "When is my appointment?",
        "Show me my schedule"
      ],
      command: [
        "Take a screenshot",
        "Open Chrome",
        "Close the window"
      ],
      question: [
        "What is the capital of France?",
        "How does this work?",
        "Why is the sky blue?"
      ],
      greeting: [
        "Hello",
        "Good morning",
        "Hi there"
      ],
      context: [
        "What did we discuss earlier?",
        "What was I saying?",
        "Remind me of our conversation"
      ]
    };
    
    this.seedEmbeddings = {};
    
    for (const [intent, examples] of Object.entries(seedExamples)) {
      this.seedEmbeddings[intent] = [];
      
      for (const example of examples) {
        const embedding = await this.generateEmbedding(example);
        this.seedEmbeddings[intent].push(embedding);
      }
    }
  }

  async generateEmbedding(text) {
    const output = await this.embedder(text, {
      pooling: 'mean',
      normalize: true
    });
    
    return Array.from(output.data);
  }

  async extractEntities(message) {
    try {
      const nerResults = await this.nerClassifier(message);
      
      const entities = nerResults.map(entity => ({
        type: this.mapEntityType(entity.entity_group),
        value: entity.word,
        entity_type: entity.entity_group,
        confidence: entity.score,
        start: entity.start,
        end: entity.end
      }));
      
      return entities;
    } catch (error) {
      console.warn('Entity extraction failed:', error.message);
      return [];
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

  async parse(message, options = {}) {
    if (!this.initialized) {
      await this.initialize();
    }

    const startTime = Date.now();
    
    try {
      // 1. Zero-shot classification
      const zeroShotResult = await this.zeroShotClassifier(message, this.candidateLabels);
      
      // 2. Embedding-based similarity
      const messageEmbedding = await this.generateEmbedding(message);
      const embeddingScores = this.calculateEmbeddingScores(messageEmbedding);
      
      // 3. Extract entities
      const entities = await this.extractEntities(message);
      
      // 4. Combine scores
      const combinedScores = this.combineScores(zeroShotResult, embeddingScores, entities, message);
      
      // 5. Get top intent
      const intent = this.getTopIntent(combinedScores);
      const confidence = combinedScores[intent];
      
      const processingTime = Date.now() - startTime;
      
      return {
        intent,
        confidence,
        entities,
        suggestedResponse: null,
        parser: 'original',
        metadata: {
          processingTimeMs: processingTime,
          zeroShotScores: this.mapZeroShotToIntents(zeroShotResult),
          embeddingScores,
          combinedScores
        }
      };
    } catch (error) {
      console.error('Original parser failed:', error);
      throw error;
    }
  }

  calculateEmbeddingScores(messageEmbedding) {
    const scores = {};
    
    for (const [intent, embeddings] of Object.entries(this.seedEmbeddings)) {
      const similarities = embeddings.map(seedEmbedding =>
        MathUtils.cosineSimilarity(messageEmbedding, seedEmbedding)
      );
      
      scores[intent] = Math.max(...similarities);
    }
    
    return scores;
  }

  mapZeroShotToIntents(zeroShotResult) {
    const mapping = {
      'storing information': 'memory_store',
      'retrieving information': 'memory_retrieve',
      'executing a command': 'command',
      'asking a question': 'question',
      'greeting': 'greeting',
      'asking about conversation history': 'context'
    };
    
    const scores = {};
    
    for (let i = 0; i < zeroShotResult.labels.length; i++) {
      const label = zeroShotResult.labels[i];
      const score = zeroShotResult.scores[i];
      const intent = mapping[label];
      
      if (intent) {
        scores[intent] = score;
      }
    }
    
    return scores;
  }

  combineScores(zeroShotResult, embeddingScores, entities, message) {
    const zeroShotScores = this.mapZeroShotToIntents(zeroShotResult);
    const combinedScores = {};
    
    // Weighted combination: 60% zero-shot, 40% embedding
    for (const intent of this.intentLabels) {
      const zeroShot = zeroShotScores[intent] || 0;
      const embedding = embeddingScores[intent] || 0;
      
      combinedScores[intent] = (zeroShot * 0.6) + (embedding * 0.4);
    }
    
    // Apply entity-based boosting
    this.applyEntityBoosting(combinedScores, entities, message);
    
    return combinedScores;
  }

  applyEntityBoosting(scores, entities, message) {
    const lowerMessage = message.toLowerCase();
    
    // Boost memory_store if has entities and memory keywords
    if (entities.length > 0 && lowerMessage.match(/remember|save|note|don't forget/)) {
      scores.memory_store *= 1.15;
    }
    
    // Boost memory_retrieve if asking questions with entities
    if (entities.length > 0 && lowerMessage.match(/^(what|when|where|who)/)) {
      scores.memory_retrieve *= 1.1;
    }
    
    // Boost command if starts with action verb
    if (lowerMessage.match(/^(open|close|launch|take|start|stop)/)) {
      scores.command *= 1.2;
    }
    
    // Boost question if ends with question mark
    if (message.trim().endsWith('?')) {
      scores.question *= 1.1;
    }
    
    // Normalize
    const maxScore = Math.max(...Object.values(scores));
    if (maxScore > 1) {
      for (const intent in scores) {
        scores[intent] = scores[intent] / maxScore;
      }
    }
  }

  getTopIntent(scores) {
    let topIntent = 'question';
    let topScore = 0;
    
    for (const [intent, score] of Object.entries(scores)) {
      if (score > topScore) {
        topScore = score;
        topIntent = intent;
      }
    }
    
    return topIntent;
  }
}

module.exports = NaturalLanguageIntentParser;
