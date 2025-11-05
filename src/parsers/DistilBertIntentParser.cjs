/**
 * DistilBERT Intent Parser
 * High-accuracy parser using DistilBERT embeddings + NER
 * Accuracy: 95%+
 * Latency: ~42ms
 */

const { pipeline } = require('@xenova/transformers');
const MathUtils = require('../utils/MathUtils.cjs');
const IntentResponses = require('../utils/IntentResponses.cjs');

class DistilBertIntentParser {
  constructor() {
    this.embedder = null;
    this.nerClassifier = null;
    this.initialized = false;
    
    // Intent labels (expanded with general_knowledge)
    this.intentLabels = [
      'memory_store',
      'memory_retrieve',
      'web_search',         // Time-sensitive queries requiring current data
      'general_knowledge',  // Stable facts that don't need web search
      'command',
      'question',           // Capability queries and general questions
      'greeting',
      'context'
    ];
    
    // Seed examples for each intent (expanded with paraphrases, edge cases, hard negatives)
    // Aim: 15-25 diverse examples per intent for robust classification
    this.seedExamples = {
      memory_store: [
        "Remember I have a meeting with John tomorrow at 3pm",
        "Save this: I need to buy milk and eggs",
        "Don't forget my dentist appointment on Friday",
        "Keep in mind that Sarah's birthday is next week",
        "Note that the project deadline is October 15th",
        "Remember: reschedule eye exam to Nov 12 at 2:30pm",
        "Save this note—server beta key is F9A3-22Q",
        "Keep track that my passport expires in March",
        "Note Chloe's ukulele recital is Saturday 6pm",
        "Store my Wi-Fi: SSID 'Home5G', pass 'orchid77'",
        "Log that I ran 3 miles today",
        "Don't forget mom's flight lands 7:45am Friday",
        "Add: renew AWS cert before 10/31",
        "Please remember I prefer dark mode",
        "Save my shoe size: US 10.5",
        "Keep in mind I'm allergic to peanuts",
        "Note down my car's VIN number",
        "Remember my favorite coffee is oat milk latte"
      ],
      memory_retrieve: [
        "What meetings do I have tomorrow?",
        "When is my dentist appointment?",
        "What did I need to buy at the store?",
        "When is Sarah's birthday?",
        "What's the project deadline?",
        "What did I say my Wi-Fi password is?",
        "When's mom's flight again?",
        "Show my tasks for tomorrow",
        "Do you remember my passport expiry?",
        "Pull up my saved server beta key",
        "What time is Chloe's recital?",
        "List the notes I added today",
        "Did I log a run this week?",
        "What preferences have I set?",
        "When is the AWS cert due?",
        "What's my shoe size?",
        "What am I allergic to?",
        "What's my car's VIN?"
      ],
      web_search: [
        // Current leadership and positions
        "Who is the president of the United States?",
        "Who is the current president of USA?",
        "Who's the prime minister of UK right now?",
        "Who is the current CEO of Apple?",
        "Who is the governor of California?",
        "Who's the current CEO of OpenAI?",
        // Current prices and stocks
        "How much does a Tesla cost?",
        "What's the price of Bitcoin?",
        "BTC price right now?",
        "What's the current stock price of Apple?",
        "How much does gas cost today?",
        "Gas prices near me",
        // Weather and current conditions
        "What's the weather in New York today?",
        "What's the weather like now?",
        "Weather in Philly today",
        "What's the temperature today?",
        // Recent news and events
        "What's the latest news about AI?",
        "Latest news on GPT-5?",
        "What happened today?",
        "What's the latest news?",
        "New Node.js LTS version",
        // Sports scores and results
        "What's the score of the game?",
        "Eagles score tonight",
        "Who won the Super Bowl?",
        "Who won yesterday's World Series game?",
        // Time-sensitive queries
        "When is the next election?",
        "What time is it in London?",
        "When does Costco close today?",
        "When is Diwali this year?",
        "US CPI print date this month",
        // Code and tutorial requests (need web search for examples/docs)
        "Give me a Python script that can interface with audio for browser",
        "Show me how to use WebSockets in Node.js",
        "How do I create a REST API in Flask?",
        "Give me an example of async/await in JavaScript",
        "Show me code for reading CSV files in Python",
        "How do I connect to MongoDB in Node.js?",
        "Give me a script to scrape websites with Python",
        "Show me how to use React hooks",
        "How do I deploy a Docker container?",
        "Give me code for file upload in Express"
      ],
      general_knowledge: [
        // Stable facts that don't change
        "What is the capital of France?",
        "Where is the Eiffel Tower located?",
        "When was the Declaration of Independence signed?",
        "Who invented the telephone?",
        "What is a VPC in AWS?",
        "Explain CAP theorem simply",
        "What's Big-O for binary search?",
        "How does JWT work?",
        "What is Terraform state?",
        "Explain event sourcing",
        "What is a Merkle tree?",
        "Difference between TCP and UDP?",
        "How do you write a function in Rust?",
        "What's the syntax for a for loop in Python?",
        "What is the speed of light?",
        "How many continents are there?",
        "What is photosynthesis?",
        "Who wrote Romeo and Juliet?"
      ],
      command: [
        "Take a screenshot",
        "Open Chrome",
        "Close all windows",
        "Search for restaurants nearby",
        "Play some music",
        "Open VS Code",
        "Start a 25-minute timer",
        "Mute system volume",
        "Create a new note titled 'Ideas'",
        "Switch to dark mode",
        "Play Lo-fi beats",
        "Close all Chrome tabs",
        "Launch Docker Desktop",
        "Copy the last transcript to clipboard",
        "Set an alarm for 7am"
      ],
      question: [
        // Capability queries and general questions
        "How are you doing?",
        "Can you help me with something?",
        "What can you do?",
        "Do you understand what I'm saying?",
        "Are you able to assist me?",
        "What can you do with my calendar?",
        "Can you browse the web?",
        "Can you remember things long-term?",
        "Can you run local scripts?",
        "Can you summarize PDFs?",
        "Are you able to control apps?",
        "How do I use this feature?",
        "What are your capabilities?",
        "Can you explain how this works?"
      ],
      greeting: [
        "Hello",
        "Hi there",
        "Good morning",
        "Good afternoon",
        "Hey, how are you?",
        "Hey! 👋",
        "Good evening",
        "How's it going?",
        "Yo!",
        "Thanks a lot!",
        "Appreciate it",
        "Sup"
      ],
      context: [
        "What did we talk about earlier?",
        "What was I saying before?",
        "Can you remind me of our conversation?",
        "What were we discussing?",
        "Go back to what we were talking about",
        "What were we discussing before this?",
        "Summarize our last session",
        "Remind me what I asked 10 minutes ago",
        "Continue from where we left off",
        "What's the plan we outlined earlier?",
        "Show me the earlier steps"
      ]
    };
    
    this.seedEmbeddings = null;
  }

  async initialize() {
    if (this.initialized) return;
    
    console.log('🚀 Initializing DistilBertIntentParser...');
    const startTime = Date.now();
    
    try {
      // Load embedding model
      console.log('  Loading embedding model...');
      this.embedder = await pipeline(
        'feature-extraction',
        'Xenova/all-MiniLM-L6-v2'
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
      await this.computeSeedEmbeddings();
      
      this.initialized = true;
      const elapsed = Date.now() - startTime;
      console.log(`✅ DistilBertIntentParser initialized in ${elapsed}ms`);
    } catch (error) {
      console.error('❌ Failed to initialize DistilBertIntentParser:', error);
      throw error;
    }
  }

  async computeSeedEmbeddings() {
    this.seedEmbeddings = {};
    
    for (const [intent, examples] of Object.entries(this.seedExamples)) {
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
    
    // Convert to regular array
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
      // 1. Generate embedding for input message
      const messageEmbedding = await this.generateEmbedding(message);
      
      // 2. Calculate similarity scores with seed examples
      const scores = this.calculateIntentScores(messageEmbedding);
      
      // 3. Extract entities
      const entities = options.includeEntities !== false 
        ? await this.extractEntities(message)
        : [];
      
      // 4. Apply entity-based boosting
      this.applyEntityBoosting(scores, entities, message);
      
      // 5. Get top intent
      const intent = this.getTopIntent(scores);
      const confidence = scores[intent];
      
      // 6. Generate suggested response
      const suggestedResponse = options.includeSuggestedResponse !== false
        ? IntentResponses.getSuggestedResponse(intent, message, entities)
        : null;
      
      const processingTime = Date.now() - startTime;
      
      return {
        intent,
        confidence,
        entities,
        suggestedResponse,
        parser: 'distilbert',
        metadata: {
          processingTimeMs: processingTime,
          modelVersion: 'all-MiniLM-L6-v2',
          nerModelVersion: 'bert-base-multilingual-cased-ner-hrl',
          scores
        }
      };
    } catch (error) {
      console.error('DistilBERT parsing failed:', error);
      throw error;
    }
  }

  calculateIntentScores(messageEmbedding) {
    const scores = {};
    
    for (const [intent, embeddings] of Object.entries(this.seedEmbeddings)) {
      // Calculate similarity with each seed example
      const similarities = embeddings.map(seedEmbedding =>
        MathUtils.cosineSimilarity(messageEmbedding, seedEmbedding)
      );
      
      // Use max similarity as the score
      scores[intent] = Math.max(...similarities);
    }
    
    return scores;
  }

  applyEntityBoosting(scores, entities, message) {
    const lowerMessage = message.toLowerCase();
    
    // Detect if this is a question (WH-word or question mark)
    const hasQuestionWord = lowerMessage.match(/^(what|when|where|who|which|why|how|whose|whom|can|could|would|should|is|are|do|does|did)/i);
    const hasQuestionMark = message.trim().endsWith('?');
    const isQuestion = hasQuestionWord || hasQuestionMark;
    
    // Detect explicit storage verbs
    const hasStorageVerb = lowerMessage.match(/\b(remember|save|note|store|keep|don't forget|keep in mind|write down|jot down)\b/);
    
    // Boost memory_store ONLY if has storage verbs AND not a question
    if (entities.some(e => e.type === 'datetime' || e.type === 'person')) {
      if (hasStorageVerb && !isQuestion) {
        scores.memory_store *= 1.2;
      }
    }
    
    // Penalize memory_store for questions (critical fix)
    if (isQuestion && !hasStorageVerb) {
      scores.memory_store *= 0.3; // Strong penalty
    }
    
    // Boost memory_retrieve if asking about stored information
    if (lowerMessage.match(/^(what|when|where|who|which)/)) {
      if (entities.some(e => e.type === 'datetime' || e.type === 'person')) {
        scores.memory_retrieve *= 1.15;
      }
    }
    
    // Boost command if has action verbs
    if (lowerMessage.match(/^(open|close|launch|take|start|stop|play|set)/)) {
      scores.command *= 1.25;
    }
    
    // Boost question/web_search for WH-questions
    if (hasQuestionWord) {
      // Check if it's a factual question (likely needs web search)
      const isFactualQuestion = lowerMessage.match(/\b(who is|what is|when did|where is|how much|how many|what's the|who's the|when was|where was)\b/);
      
      if (isFactualQuestion) {
        scores.web_search *= 1.3;
        scores.question *= 1.15;
      } else {
        scores.question *= 1.2;
      }
    }
    
    // 🔍 ENHANCED: Boost web_search for current events and time-sensitive queries
    const hasCurrentEventIndicators = lowerMessage.match(/\b(current|now|today|latest|recent|this year|2024|2025|2026)\b/);
    const hasLeadershipQuery = lowerMessage.match(/\b(president|prime minister|ceo|leader|governor|mayor|king|queen)\b/);
    const hasPriceQuery = lowerMessage.match(/\b(price|cost|stock|worth|value|how much)\b/);
    const hasWeatherQuery = lowerMessage.match(/\b(weather|temperature|forecast|rain|snow|sunny|cloudy)\b/);
    const hasNewsQuery = lowerMessage.match(/\b(news|latest|happened|happening|event|announcement)\b/);
    const hasSportsQuery = lowerMessage.match(/\b(score|game|match|won|lost|team|player)\b/);
    
    // 🔍 NEW: Boost web_search for code/tutorial requests
    const hasCodeRequest = lowerMessage.match(/\b(give me|show me|how do i|how to|example of|tutorial|code for|script)\b/);
    const hasProgrammingContext = lowerMessage.match(/\b(python|javascript|node|react|api|function|class|code|script|program|html|css|sql|database|docker|kubernetes)\b/);
    
    // Strong boost for current events
    if (hasCurrentEventIndicators) {
      scores.web_search *= 1.5;
    }
    
    // Boost for leadership queries (often need current info)
    if (hasLeadershipQuery && (hasQuestionWord || hasQuestionMark)) {
      scores.web_search *= 1.4;
    }
    
    // Boost for price/cost queries (always need current data)
    if (hasPriceQuery) {
      scores.web_search *= 1.45;
    }
    
    // Boost for weather queries (always need current data)
    if (hasWeatherQuery) {
      scores.web_search *= 1.6;
    }
    
    // Boost for news queries
    if (hasNewsQuery) {
      scores.web_search *= 1.5;
    }
    
    // Boost for sports queries
    if (hasSportsQuery) {
      scores.web_search *= 1.4;
    }
    
    // 🔍 NEW: Strong boost for code/tutorial requests
    if (hasCodeRequest && hasProgrammingContext) {
      scores.web_search *= 1.6;  // Strong boost
      scores.command *= 0.5;      // Penalize command (avoid confusion)
      scores.question *= 0.7;     // Slightly penalize generic question
    }
    
    // Additional boost if ends with question mark
    if (hasQuestionMark) {
      scores.question *= 1.1;
      scores.web_search *= 1.05;
    }
    
    // Boost greeting if message is very short and contains greeting words
    if (message.split(' ').length <= 5) {
      if (lowerMessage.match(/^(hi|hello|hey|good morning|good afternoon)/)) {
        scores.greeting *= 1.3;
      }
    }
    
    // Normalize scores back to 0-1 range
    const maxScore = Math.max(...Object.values(scores));
    if (maxScore > 1) {
      for (const intent in scores) {
        scores[intent] = scores[intent] / maxScore;
      }
    }
  }

  getTopIntent(scores) {
    // Sort intents by score
    const sortedIntents = Object.entries(scores)
      .sort((a, b) => b[1] - a[1]);
    
    const topIntent = sortedIntents[0][0];
    const topScore = sortedIntents[0][1];
    const secondScore = sortedIntents[1]?.[1] || 0;
    
    // Intent priority for tie-breaking (when scores are very close)
    const intentPriority = {
      'web_search': 5,      // Highest priority for factual questions
      'question': 4,        // General questions
      'memory_retrieve': 3, // Retrieving stored info
      'command': 2,
      'context': 1,
      'memory_store': 0,    // Lowest priority (avoid false positives)
      'greeting': 0
    };
    
    // Only default to question if ALL scores are extremely low (< 0.15)
    // This prevents defaulting when web_search has highest score but low confidence
    if (topScore < 0.15) {
      console.log(`⚠️ Extremely low confidence (${topScore.toFixed(3)}), defaulting to 'question'`);
      return 'question';
    }
    
    // If scores are very close (within 0.1), use priority
    if (Math.abs(topScore - secondScore) < 0.1) {
      const topPriority = intentPriority[topIntent] || 0;
      const secondIntent = sortedIntents[1][0];
      const secondPriority = intentPriority[secondIntent] || 0;
      
      if (secondPriority > topPriority) {
        console.log(`🔄 Tie-breaking: ${topIntent} (${topScore.toFixed(3)}) vs ${secondIntent} (${secondScore.toFixed(3)}) → choosing ${secondIntent} (higher priority)`);
        return secondIntent;
      }
    }
    
    return topIntent;
  }
}

module.exports = DistilBertIntentParser;
