/**
 * LLM Service
 * Handles general Q&A using Phi4 model via Ollama API
 */

const axios = require('axios');

class LLMService {
  constructor() {
    this.initialized = false;
    this.apiUrl = process.env.PHI4_API_URL || 'http://127.0.0.1:11434/api/generate';
    this.model = process.env.PHI4_MODEL || 'phi4';
    this.enabled = process.env.ENABLE_PHI4 === 'true';
  }

  async initialize() {
    if (this.initialized) return;
    
    console.log('🚀 Initializing LLM Service...');
    
    if (!this.enabled) {
      console.log('⚠️  LLM Service disabled (ENABLE_PHI4=false)');
      this.initialized = true;
      return;
    }
    
    // Test connection to Ollama
    try {
      const testUrl = this.apiUrl.replace('/api/generate', '/api/tags');
      await axios.get(testUrl, { timeout: 2000 });
      console.log(`✅ LLM Service initialized (Ollama at ${this.apiUrl})`);
    } catch (error) {
      console.warn('⚠️  Ollama not reachable, LLM will use fallback responses');
    }
    
    this.initialized = true;
  }

  async generateAnswer(query, options = {}) {
    if (!this.initialized) {
      await this.initialize();
    }

    const startTime = Date.now();

    // Extract memories, conversation history, and system instructions from context
    const memories = options.context?.memories || [];
    const conversationHistory = options.context?.conversationHistory || [];
    const systemInstructions = options.context?.systemInstructions || '';
    
    // Log memory usage
    console.log('📚 [GENERAL-ANSWER] Received request:', {
      query: query,
      memoryCount: memories.length,
      conversationLength: conversationHistory.length,
      hasSystemInstructions: !!systemInstructions
    });
    
    if (memories.length > 0) {
      console.log('📚 [GENERAL-ANSWER] Memories being used:');
      memories.forEach((m, idx) => {
        const preview = m.text.substring(0, 50).replace(/\n/g, ' ');
        console.log(`  ${idx + 1}. "${preview}..." (similarity: ${m.similarity.toFixed(3)})`);
      });
    }

    // If LLM is disabled, return placeholder
    if (!this.enabled) {
      return this._generatePlaceholderResponse(query, options, startTime);
    }

    try {
      const prompt = this._buildPrompt(query, memories, conversationHistory, systemInstructions);
      
      const modelToUse = options.model || this.model;
      console.log(`🤖 [LLM] Using model: ${modelToUse}`);
      
      const response = await axios.post(this.apiUrl, {
        model: modelToUse,
        prompt: prompt,
        stream: false,
        options: {
          temperature: options.temperature || 0.7,  // Natural, varied responses
          num_predict: options.maxTokens || 150,    // Enough for complete thoughts
          top_k: options.topK || 40,
          top_p: options.topP || 0.9,
          num_ctx: options.contextLength || 2048,
          stop: [
            '\n\nUser:',
            '\nUser:',
            '\nUSER:',
            'USER:',
            '\nuser:',
            'user:',
            '\nAssistant:',
            '\nASSISTANT:',
            'ASSISTANT:',
            '\nassistant:',
            'assistant:',
            '**[End',
            '[End',
            '*Note',
            'Note to developers',
            '\n\n###',
            'The user asked',
            'The assistant'
          ]
        }
      }, {
        timeout: options.timeout || 30000,
        headers: { 'Content-Type': 'application/json' }
      });

      const processingTimeMs = Date.now() - startTime;
      
      return {
        answer: response.data.response.trim(),
        confidence: this._calculateConfidence(response.data),
        sources: options.sources || [],
        tokensUsed: response.data.eval_count || 0,
        usedMemories: memories.length > 0,
        memoryCount: memories.length,
        metadata: {
          model: response.data.model || this.model,
          processingTimeMs,
          temperature: options.temperature || 0.7,
          finishReason: response.data.done ? 'complete' : 'incomplete',
          totalDuration: response.data.total_duration,
          loadDuration: response.data.load_duration,
          promptEvalCount: response.data.prompt_eval_count,
          evalCount: response.data.eval_count
        }
      };
    } catch (error) {
      console.error('Answer generation failed:', error.message);
      
      // Fallback to placeholder if Ollama is unavailable
      if (error.code === 'ECONNREFUSED' || error.code === 'ETIMEDOUT') {
        console.warn('Ollama unavailable, returning placeholder response');
        return this._generatePlaceholderResponse(query, options, startTime);
      }
      
      throw new Error(`Answer generation failed: ${error.message}`);
    }
  }

  _buildPrompt(query, memories, conversationHistory, systemInstructions = '') {
    // Start with system instructions if provided (highest priority)
    let prompt = '';
    
    if (systemInstructions) {
      prompt += `${systemInstructions}\n\n`;
      console.log('📋 [GENERAL-ANSWER] Using system instructions from orchestrator');
    }
    
    // Enhanced system prompt with explicit memory usage instructions
    prompt += `You are an AI assistant helping a user. You have access to multiple context layers:

1. **conversationHistory**: Recent back-and-forth messages (use for immediate context)
2. **memories**: Long-term factual information about the user from previous conversations (use for questions about user preferences, history, or past statements)

CRITICAL RULES:
- Always respond from the ASSISTANT's perspective (use "you" for the user, never "I")
- When the user asks "what do I like/love/prefer" or "what is my favorite", CHECK THE MEMORIES FIRST
- If memories exist and are relevant, USE THEM in your response
- If NO memories are provided or memories don't contain the answer, say "I don't have that information stored yet"
- NEVER make up or invent information that isn't in the memories or conversation history
- NEVER make negative inferences: If user says "I love X", DO NOT assume they dislike Y
- DO NOT infer preferences from unrelated memories: "loves roses" does NOT mean "dislikes dandelions"
- Only state facts that are EXPLICITLY mentioned - do not infer, assume, or extrapolate
- Answer naturally in 1-2 sentences
- Do not simulate conversations or add role labels

`;

    // Check if this is a factual query about user preferences
    const isFactualQuery = /\b(what|my|favorite|like|love|prefer|do i)\b/i.test(query);
    
    // 1. Add system messages from conversation history (highest priority)
    if (conversationHistory && conversationHistory.length > 0) {
      const systemMessages = conversationHistory.filter(m => m.role === 'system');
      if (systemMessages.length > 0) {
        console.log(`📚 [GENERAL-ANSWER] Found ${systemMessages.length} system messages`);
        systemMessages.forEach(msg => {
          prompt += `${msg.content}\n\n`;
        });
      }
    }
    
    // 2. Add memory context if relevant (especially for factual queries)
    if (memories && memories.length > 0) {
      if (isFactualQuery) {
        prompt += 'RELEVANT MEMORIES ABOUT THE USER:\n';
        memories.forEach((m, idx) => {
          // Extract the user's statement from memory text
          const memoryText = m.text.replace(/User asked: "|Assistant responded: "/g, '').split('\n')[0];
          prompt += `${idx + 1}. ${memoryText}\n`;
        });
        prompt += '\nUse these memories to answer the user\'s question. Speak from the assistant\'s perspective (use "you" for the user).\n\n';
        console.log(`📚 [GENERAL-ANSWER] Using ${memories.length} memories for factual query`);
      } else {
        // For non-factual queries, still include memories but less prominently
        prompt += 'Context about the user:\n';
        memories.forEach((m) => {
          const memoryText = m.text.replace(/User asked: "|Assistant responded: "/g, '').split('\n')[0];
          prompt += `- ${memoryText}\n`;
        });
        prompt += '\n';
        console.log(`📚 [GENERAL-ANSWER] Using ${memories.length} memories as background context`);
      }
    } else {
      // Explicitly tell the model there are NO memories
      if (isFactualQuery) {
        prompt += 'NO MEMORIES AVAILABLE: There are no stored memories about this topic. If the user asks about their preferences or past statements, respond: "I don\'t have that information stored yet."\n\n';
        console.log('⚠️ [GENERAL-ANSWER] No memories provided - instructing model to admit lack of knowledge');
      } else {
        console.log('⚠️ [GENERAL-ANSWER] No memories provided');
      }
    }
    
    // 3. Add conversation history (excluding system messages)
    if (conversationHistory && conversationHistory.length > 0) {
      const dialogue = conversationHistory.filter(m => m.role !== 'system');
      const recentHistory = dialogue.slice(-8); // Last 8 messages for better context
      const cleanedHistory = recentHistory.map(c => ({
        role: c.role,
        // Remove any [End] markers and meta-commentary from history
        content: c.content
          .replace(/\*\*\[End.*?\]\*\*/g, '')
          .replace(/\[End.*?\]/g, '')
          .replace(/\*Note.*?\*/g, '')
          .replace(/Note to developers.*$/s, '')
          .replace(/USER:.*$/gm, '')
          .replace(/ASSISTANT:.*$/gm, '')
          .replace(/The user asked.*$/gm, '')
          .replace(/The assistant.*$/gm, '')
          .trim()
      })).filter(c => c.content.length > 0); // Remove empty messages
      
      if (cleanedHistory.length > 0) {
        prompt += 'CONVERSATION:\n';
        cleanedHistory.forEach(c => {
          const label = c.role === 'user' ? 'USER' : 'ASSISTANT';
          prompt += `${label}: ${c.content}\n`;
        });
        prompt += '\n';
      }
    }
    
    // 4. Add current query with clear instruction
    prompt += `USER: ${query}\nASSISTANT:`;
    
    console.log('📚 [GENERAL-ANSWER] System prompt length:', prompt.length);
    console.log('📚 [GENERAL-ANSWER] Is factual query:', isFactualQuery);
    
    return prompt;
  }

  _calculateConfidence(ollamaResponse) {
    // Simple confidence calculation based on response completeness
    if (ollamaResponse.done && ollamaResponse.response) {
      return 0.85; // High confidence for complete responses
    }
    return 0.5;
  }

  _generatePlaceholderResponse(query, options, startTime) {
    const processingTimeMs = Date.now() - startTime;
    
    return {
      answer: `This is a placeholder response for: "${query}". ` +
        `To enable Phi4 LLM, set ENABLE_PHI4=true and ensure Ollama is running at ${this.apiUrl}. ` +
        `Install with: ollama pull phi4`,
      confidence: 0.5,
      sources: [],
      tokensUsed: 0,
      metadata: {
        model: 'placeholder',
        processingTimeMs,
        temperature: options.temperature || 0.7,
        finishReason: 'placeholder',
        note: 'LLM service disabled or unavailable'
      }
    };
  }
}

module.exports = new LLMService();
