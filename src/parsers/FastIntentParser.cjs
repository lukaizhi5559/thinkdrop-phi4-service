/**
 * Fast Intent Parser
 * Lightweight parser using Natural + Compromise
 * No heavy dependencies, fast response time (~15ms)
 * Accuracy: ~82%
 */

const natural = require('natural');
const nlp = require('compromise');

class FastIntentParser {
  constructor() {
    this.stemmer = natural.PorterStemmer;
    this.initialized = false;
    
    // Intent patterns
    this.patterns = {
      memory_store: {
        keywords: ['remember', 'save', 'store', 'keep', 'note', 'don\'t forget', 'remind me to'],
        verbs: ['have', 'need', 'must', 'should', 'promised'],
        weight: 1.0
      },
      memory_retrieve: {
        keywords: ['when', 'what', 'where', 'who', 'which', 'show', 'find', 'search', 'tell me', 'remind me of'],
        verbs: ['is', 'are', 'was', 'were', 'have', 'do', 'did'],
        weight: 0.9
      },
      command: {
        keywords: ['open', 'close', 'launch', 'start', 'stop', 'take', 'capture', 'screenshot', 'play', 'pause', 'set', 'turn'],
        verbs: ['open', 'close', 'launch', 'start', 'stop', 'take', 'capture', 'play', 'set'],
        weight: 1.0
      },
      question: {
        keywords: ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'explain', 'tell me about'],
        patterns: [/^(what|how|why|when|where|who|which)/i, /\?$/],
        weight: 0.8
      },
      greeting: {
        keywords: ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'greetings', 'howdy', 'sup', 'yo'],
        weight: 0.7
      },
      context: {
        keywords: ['earlier', 'before', 'previous', 'last', 'conversation', 'discussed', 'talked about', 'said', 'mentioned'],
        weight: 0.8
      }
    };
  }

  async initialize() {
    if (this.initialized) return;
    console.log('🚀 FastIntentParser initialized');
    this.initialized = true;
  }

  async parse(message, options = {}) {
    if (!this.initialized) {
      await this.initialize();
    }

    const startTime = Date.now();
    
    // Normalize message
    const lowerMessage = message.toLowerCase().trim();
    const doc = nlp(message);
    
    // Calculate scores for each intent
    const scores = {};
    
    // Memory Store Detection
    scores.memory_store = this.scoreMemoryStore(doc, lowerMessage);
    
    // Memory Retrieve Detection
    scores.memory_retrieve = this.scoreMemoryRetrieve(doc, lowerMessage);
    
    // Command Detection
    scores.command = this.scoreCommand(doc, lowerMessage);
    
    // Question Detection
    scores.question = this.scoreQuestion(doc, lowerMessage);
    
    // Greeting Detection
    scores.greeting = this.scoreGreeting(doc, lowerMessage);
    
    // Context Detection
    scores.context = this.scoreContext(doc, lowerMessage);
    
    // Get top intent
    const intent = this.getTopIntent(scores);
    const confidence = scores[intent] || 0.5;
    
    const processingTime = Date.now() - startTime;
    
    return {
      intent,
      confidence,
      entities: [], // Fast parser doesn't extract entities
      suggestedResponse: null,
      parser: 'fast',
      metadata: {
        processingTimeMs: processingTime,
        scores
      }
    };
  }

  scoreMemoryStore(doc, lowerMessage) {
    let score = 0;
    const pattern = this.patterns.memory_store;
    
    // Check for memory keywords
    for (const keyword of pattern.keywords) {
      if (lowerMessage.includes(keyword)) {
        score += 0.3;
      }
    }
    
    // Check for future tense or temporal markers
    if (doc.has('#FutureTime') || doc.has('#Date') || doc.has('#Time')) {
      score += 0.2;
    }
    
    // Check for action verbs
    const verbs = doc.verbs().out('array');
    for (const verb of verbs) {
      if (pattern.verbs.includes(verb.toLowerCase())) {
        score += 0.15;
      }
    }
    
    // Check for imperative mood (commands to remember)
    if (doc.has('#Imperative')) {
      score += 0.2;
    }
    
    // Boost if contains person names or places
    if (doc.has('#Person') || doc.has('#Place')) {
      score += 0.15;
    }
    
    return Math.min(score, 1.0);
  }

  scoreMemoryRetrieve(doc, lowerMessage) {
    let score = 0;
    const pattern = this.patterns.memory_retrieve;
    
    // Check for question words
    for (const keyword of pattern.keywords) {
      if (lowerMessage.startsWith(keyword) || lowerMessage.includes(keyword)) {
        score += 0.3;
      }
    }
    
    // Check if it's a question
    if (doc.has('#Question') || lowerMessage.includes('?')) {
      score += 0.25;
    }
    
    // Check for retrieval verbs
    const verbs = doc.verbs().out('array');
    for (const verb of verbs) {
      if (pattern.verbs.includes(verb.toLowerCase())) {
        score += 0.1;
      }
    }
    
    // Check for temporal references (asking about past/future)
    if (doc.has('#Date') || doc.has('#Time') || lowerMessage.includes('when')) {
      score += 0.2;
    }
    
    // Penalize if it's clearly a store command
    if (lowerMessage.includes('remember') && !lowerMessage.includes('?')) {
      score -= 0.3;
    }
    
    return Math.max(0, Math.min(score, 1.0));
  }

  scoreCommand(doc, lowerMessage) {
    let score = 0;
    const pattern = this.patterns.command;
    
    // Check for command keywords
    for (const keyword of pattern.keywords) {
      if (lowerMessage.includes(keyword)) {
        score += 0.4;
      }
    }
    
    // Check for imperative verbs
    const verbs = doc.verbs().out('array');
    for (const verb of verbs) {
      if (pattern.verbs.includes(verb.toLowerCase())) {
        score += 0.3;
      }
    }
    
    // Check for imperative mood
    if (doc.has('#Imperative')) {
      score += 0.2;
    }
    
    // Check for application names
    if (lowerMessage.match(/chrome|firefox|safari|spotify|slack|vscode|terminal/i)) {
      score += 0.2;
    }
    
    // Short imperative sentences are likely commands
    if (lowerMessage.split(' ').length <= 4 && doc.has('#Imperative')) {
      score += 0.15;
    }
    
    return Math.min(score, 1.0);
  }

  scoreQuestion(doc, lowerMessage) {
    let score = 0;
    const pattern = this.patterns.question;
    
    // Check for question keywords
    for (const keyword of pattern.keywords) {
      if (lowerMessage.startsWith(keyword)) {
        score += 0.35;
      }
    }
    
    // Check if marked as question
    if (doc.has('#Question')) {
      score += 0.3;
    }
    
    // Check for question mark
    if (lowerMessage.includes('?')) {
      score += 0.25;
    }
    
    // Check for question patterns
    for (const regex of pattern.patterns) {
      if (regex.test(lowerMessage)) {
        score += 0.2;
      }
    }
    
    // Penalize if it's asking about personal information (likely memory_retrieve)
    if (lowerMessage.match(/my|i have|i need|appointment|meeting|schedule/i)) {
      score -= 0.3;
    }
    
    return Math.max(0, Math.min(score, 1.0));
  }

  scoreGreeting(doc, lowerMessage) {
    let score = 0;
    const pattern = this.patterns.greeting;
    
    // Check for greeting keywords
    for (const keyword of pattern.keywords) {
      if (lowerMessage.includes(keyword)) {
        score += 0.5;
      }
    }
    
    // Short messages with greetings are likely pure greetings
    if (lowerMessage.split(' ').length <= 5 && score > 0) {
      score += 0.3;
    }
    
    // Check for greeting patterns
    if (lowerMessage.match(/^(hi|hello|hey|good (morning|afternoon|evening))/i)) {
      score += 0.4;
    }
    
    return Math.min(score, 1.0);
  }

  scoreContext(doc, lowerMessage) {
    let score = 0;
    const pattern = this.patterns.context;
    
    // Check for context keywords
    for (const keyword of pattern.keywords) {
      if (lowerMessage.includes(keyword)) {
        score += 0.35;
      }
    }
    
    // Check for past tense
    if (doc.has('#PastTense')) {
      score += 0.2;
    }
    
    // Check for conversation-related words
    if (lowerMessage.match(/we|our|conversation|chat|discuss|talk|said|told/i)) {
      score += 0.2;
    }
    
    // Check for question about previous interaction
    if (lowerMessage.match(/what (did|was|were)|remind me (of|about)/i)) {
      score += 0.25;
    }
    
    return Math.min(score, 1.0);
  }

  getTopIntent(scores) {
    let topIntent = 'question'; // Default fallback
    let topScore = 0;
    
    for (const [intent, score] of Object.entries(scores)) {
      if (score > topScore) {
        topScore = score;
        topIntent = intent;
      }
    }
    
    // If all scores are very low, default to question
    if (topScore < 0.3) {
      return 'question';
    }
    
    return topIntent;
  }
}

module.exports = FastIntentParser;
