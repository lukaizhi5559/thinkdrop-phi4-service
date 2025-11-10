/**
 * LLM Service
 * Handles general Q&A using Phi4 model via Ollama API
 */

const axios = require('axios');

class LLMService {
  constructor() {
    this.initialized = false;
    this.apiUrl = process.env.PHI4_API_URL || 'http://127.0.0.1:11434/api/generate';
    this.model = process.env.OLLAMA_MODEL || 'qwen2:1.5b';
    this.enabled = process.env.ENABLE_PHI4 === 'true';
    
    // Performance settings from environment with optimized defaults
    this.defaultMaxTokens = parseInt(process.env.PHI4_MAX_TOKENS) || 250;
    this.defaultContextLength = parseInt(process.env.PHI4_CONTEXT_LENGTH) || 4096;
    this.defaultNumThread = parseInt(process.env.PHI4_NUM_THREAD) || 8;
    this.defaultNumGpu = parseInt(process.env.PHI4_NUM_GPU) || 1;
    this.defaultRepeatPenalty = parseFloat(process.env.PHI4_REPEAT_PENALTY) || 1.1;
    this.defaultTemperature = parseFloat(process.env.PHI4_TEMPERATURE) || 0.7;
    this.defaultTopP = parseFloat(process.env.PHI4_TOP_P) || 0.9;
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

    // Extract memories, conversation history, web results, and system instructions from context
    const memories = options.context?.memories || [];
    const conversationHistory = options.context?.conversationHistory || [];
    const webSearchResults = options.context?.webSearchResults || [];
    const systemInstructions = options.context?.systemInstructions || '';
    
    // Log context usage
    console.log('📚 [GENERAL-ANSWER] Received request:', {
      query: query,
      memoryCount: memories.length,
      conversationLength: conversationHistory.length,
      webResultsCount: webSearchResults.length,
      hasSystemInstructions: !!systemInstructions
    });
    
    if (memories.length > 0) {
      console.log('📚 [GENERAL-ANSWER] Memories being used:');
      memories.forEach((m, idx) => {
        const preview = m.text.substring(0, 50).replace(/\n/g, ' ');
        console.log(`  ${idx + 1}. "${preview}..." (similarity: ${m.similarity.toFixed(3)})`);
      });
    }

    if (webSearchResults.length > 0) {
      console.log('🌐 [GENERAL-ANSWER] Web search results being used:');
      webSearchResults.forEach((r, idx) => {
        const preview = r.text.substring(0, 60).replace(/\n/g, ' ');
        console.log(`  ${idx + 1}. "${preview}..."`);
      });
    }

    // If LLM is disabled, return placeholder
    if (!this.enabled) {
      return this._generatePlaceholderResponse(query, options, startTime);
    }

    try {
      const prompt = this._buildPrompt(query, memories, conversationHistory, webSearchResults, systemInstructions);
      
      const modelToUse = options.model || this.model;
      console.log(`🤖 [LLM] Using model: ${modelToUse}`);
      console.log(`⚡ [LLM] Performance settings: maxTokens=${this.defaultMaxTokens}, contextLength=${this.defaultContextLength}, numThread=${this.defaultNumThread}`);
      
      const response = await axios.post(this.apiUrl, {
        model: modelToUse,
        prompt: prompt,
        stream: false,
        options: {
          temperature: options.temperature || this.defaultTemperature,
          num_predict: options.maxTokens || this.defaultMaxTokens,
          top_k: options.topK || 40,
          top_p: options.topP || this.defaultTopP,
          num_ctx: options.contextLength || this.defaultContextLength,
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

  _buildPrompt(query, memories, conversationHistory, webSearchResults = [], systemInstructions = '') {
    // Start with system instructions if provided (highest priority)
    let prompt = '';
    
    if (systemInstructions) {
      prompt += `${systemInstructions}\n\n`;
      console.log('📋 [GENERAL-ANSWER] Using system instructions from orchestrator');
      // If custom system instructions provided, skip generic instructions
      // This is critical for screen_intelligence and vision intents
    } else {
      // Only add generic instructions if no custom instructions provided
      // Enhanced system prompt with explicit memory usage instructions
      prompt += `You are an AI assistant helping a user. You have access to multiple context layers:

1. **webSearchResults**: Current information from the web (use for factual questions about the world)
2. **conversationHistory**: Recent back-and-forth messages (use for immediate context and to recall what was discussed)
3. **memories**: Long-term factual information about the user from previous conversations (use for questions about user preferences, history, or past statements)

CRITICAL RULES:
- Always respond from the ASSISTANT's perspective (use "you" for the user, never "I")
- **PRONOUN RESOLUTION**: When the user uses pronouns like "he", "she", "it", "they", "him", "her", FIRST check the conversation history to identify who/what is being referenced BEFORE using web search results
- When the user asks "what did we talk about" or "what have we discussed", REVIEW THE CONVERSATION HISTORY and summarize the topics
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
    }

    // For screen_intelligence and vision intents, skip adding extra context
    // The query already contains all necessary context
    const hasCustomInstructions = !!systemInstructions;
    
    // Check if this is a factual query
    // BEST INDICATOR: If web search results are present, it's DEFINITELY a factual query
    // Otherwise, check if it's a user preference query using patterns
    const hasWebResults = webSearchResults && webSearchResults.length > 0;
    const isUserPreferenceQuery = /\b(what|my|favorite|like|love|prefer|do i)\b/i.test(query);
    const isFactualQuery = hasWebResults || isUserPreferenceQuery;
    
    // Only add context if no custom instructions (for screen_intelligence, query has everything)
    if (!hasCustomInstructions) {
    // 1. Add web search results if available (highest priority for factual questions)
    // ULTRA-OPTIMIZED: Only use top 1 result, 120 chars for fastest processing
    if (webSearchResults && webSearchResults.length > 0) {
      prompt += 'INFO: ';
      const truncated = webSearchResults[0].text.substring(0, 120);
      prompt += `${truncated}...\n\n`;
      console.log(`🌐 [GENERAL-ANSWER] Using 1 web result (120 chars)`);
    }
    
    // 2. Add system messages from conversation history
    if (conversationHistory && conversationHistory.length > 0) {
      const systemMessages = conversationHistory.filter(m => m.role === 'system');
      if (systemMessages.length > 0) {
        console.log(`📚 [GENERAL-ANSWER] Found ${systemMessages.length} system messages`);
        systemMessages.forEach(msg => {
          prompt += `${msg.content}\n\n`;
        });
      }
    }
    
    // 3. Add memory context if relevant (especially for factual queries about the user)
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
      // BUT: Only add "search online" instruction if there are ALSO no web results
      if (isFactualQuery) {
        if (webSearchResults && webSearchResults.length > 0) {
          // Web results are available, so don't suggest searching
          prompt += 'NO MEMORIES AVAILABLE: There are no stored memories about the user.\n\nIMPORTANT: Use the web search results provided above to answer factual questions.\n\n';
          console.log('⚠️ [GENERAL-ANSWER] No memories, but web results available - instructing to use web results');
        } else {
          // No web results AND no memories - suggest searching
          prompt += 'NO MEMORIES AVAILABLE: There are no stored memories about the user.\n\nIMPORTANT DISTINCTION:\n- If the user asks about THEIR OWN preferences/past statements (e.g., "what do I like", "what did I say"), respond: "I don\'t have that information stored yet."\n- If the user asks about FACTUAL INFORMATION about the world (e.g., "who is X", "what is Y", "how old is Z"), respond: "I need to search online for that information. Let me look that up for you."\n\n';
          console.log('⚠️ [GENERAL-ANSWER] No memories or web results - instructing model to distinguish personal vs factual queries');
        }
      } else {
        console.log('⚠️ [GENERAL-ANSWER] No memories provided');
      }
    }
    
    // 3. Add conversation history with hybrid approach:
    //    - Always include last 6 messages (3 turns) for immediate context
    //    - Include up to 8 additional relevant messages if they mention similar topics
    if (conversationHistory && conversationHistory.length > 0) {
      const dialogue = conversationHistory.filter(m => m.role !== 'system');
      
      // Keep last 6 messages (3 turns) for context awareness
      const recentMessages = dialogue.slice(-6);
      
      // Get older messages (if any) for relevance filtering
      const olderMessages = dialogue.slice(0, -6);
      
      // Simple keyword-based relevance: check if older messages share keywords with current query
      const queryKeywords = query.toLowerCase().split(/\s+/).filter(w => w.length > 3);
      const relevantOlderMessages = olderMessages
        .filter(m => {
          const content = m.content.toLowerCase();
          // Message is relevant if it shares 2+ keywords with current query
          const sharedKeywords = queryKeywords.filter(kw => content.includes(kw));
          return sharedKeywords.length >= 2;
        })
        .slice(-8); // Max 8 additional relevant messages
      
      // Combine: relevant older messages + recent messages
      const contextMessages = [...relevantOlderMessages, ...recentMessages];
      
      const cleanedHistory = contextMessages.map(c => ({
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
        console.log(`📚 [GENERAL-ANSWER] Using ${cleanedHistory.length} messages (${recentMessages.length} recent + ${relevantOlderMessages.length} relevant older)`);
      }
    }
    } // End if (!hasCustomInstructions)
    
    // 4. Add current query with clear instruction
    prompt += `USER: ${query}\nASSISTANT:`;
    
    console.log('📚 [GENERAL-ANSWER] System prompt length:', prompt.length);
    console.log('📚 [GENERAL-ANSWER] Is factual query:', isFactualQuery);
    
    return prompt;
  }

  async generateAnswerStream(query, options = {}, onToken) {
    if (!this.initialized) {
      await this.initialize();
    }

    // Extract context same as non-streaming
    const memories = options.context?.memories || [];
    const conversationHistory = options.context?.conversationHistory || [];
    const webSearchResults = options.context?.webSearchResults || [];
    const systemInstructions = options.context?.systemInstructions || '';
    
    console.log('🌊 [STREAM] Starting streaming response for:', query.substring(0, 50));

    if (!this.enabled) {
      // Send placeholder response token by token for testing
      const placeholder = `This is a placeholder streaming response for: "${query}".`;
      for (const char of placeholder) {
        onToken({ token: char });
        await new Promise(resolve => setTimeout(resolve, 10)); // Simulate streaming
      }
      return;
    }

    try {
      const prompt = this._buildPrompt(query, memories, conversationHistory, webSearchResults, systemInstructions);
      const modelToUse = options.model || this.model;

      // Make streaming request to Ollama
      const response = await axios.post(this.apiUrl, {
        model: modelToUse,
        prompt: prompt,
        stream: true, // ⚡ Enable streaming
        options: {
          temperature: options.temperature || this.defaultTemperature,
          num_predict: options.maxTokens || this.defaultMaxTokens,
          top_k: options.topK || 40,
          top_p: options.topP || this.defaultTopP,
          num_ctx: options.contextLength || this.defaultContextLength,
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
        responseType: 'stream',
        timeout: options.timeout || 60000
      });

      // Process stream line by line
      return new Promise((resolve, reject) => {
        let buffer = '';

        response.data.on('data', (chunk) => {
          buffer += chunk.toString();
          const lines = buffer.split('\n');
          
          // Keep last incomplete line in buffer
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (line.trim()) {
              try {
                const data = JSON.parse(line);
                if (data.response) {
                  // Send token to callback
                  onToken({ token: data.response });
                  console.log('🌊 [STREAM] Token:', data.response.substring(0, 20));
                }
                
                if (data.done) {
                  console.log('✅ [STREAM] Streaming complete');
                  resolve();
                }
              } catch (err) {
                console.warn('⚠️ [STREAM] Failed to parse line:', line);
              }
            }
          }
        });

        response.data.on('end', () => {
          resolve();
        });

        response.data.on('error', (err) => {
          console.error('❌ [STREAM] Stream error:', err);
          reject(err);
        });
      });

    } catch (error) {
      console.error('❌ [STREAM] Streaming failed:', error.message);
      
      // Fallback: send error message as stream
      const errorMsg = `Error generating response: ${error.message}`;
      for (const char of errorMsg) {
        onToken({ token: char });
        await new Promise(resolve => setTimeout(resolve, 10));
      }
    }
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
