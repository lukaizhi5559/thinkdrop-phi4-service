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

    // Extract memories and conversation history from context
    const memories = options.context?.memories || [];
    const conversationHistory = options.context?.conversationHistory || [];
    
    // Log memory usage
    console.log('📚 [GENERAL-ANSWER] Received request:', {
      query: query,
      memoryCount: memories.length,
      conversationLength: conversationHistory.length
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
      const prompt = this._buildPrompt(query, memories, conversationHistory);
      
      const response = await axios.post(this.apiUrl, {
        model: options.model || this.model,
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

  _buildPrompt(query, memories, conversationHistory) {
    // Build concise system instructions - no role labels to avoid simulation
    let prompt = 'Answer the question naturally in 1-2 sentences. Do not simulate conversations or add role labels.\n\n';
    
    // Add memories if available
    if (memories && memories.length > 0) {
      prompt += 'Context:\n';
      memories.forEach((m, idx) => {
        // Extract just the key info from memory text
        const memoryText = m.text.replace(/User asked: "|Assistant responded: "/g, '').split('\n')[0];
        prompt += `- ${memoryText}\n`;
      });
      prompt += '\n';
      
      console.log(`📚 [GENERAL-ANSWER] Using ${memories.length} memories in prompt`);
    } else {
      console.log('⚠️ [GENERAL-ANSWER] No memories provided');
    }
    
    // Clean and add conversation history (last few exchanges only)
    if (conversationHistory && conversationHistory.length > 0) {
      const recentHistory = conversationHistory.slice(-4); // Only last 4 messages
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
        prompt += 'Recent conversation:\n';
        cleanedHistory.forEach(c => {
          const label = c.role === 'user' ? 'Q' : 'A';
          prompt += `${label}: ${c.content}\n`;
        });
        prompt += '\n';
      }
    }
    
    // Add current query with clear instruction
    prompt += `Question: ${query}\n\nAnswer (1-2 sentences only):`;
    
    console.log('📚 [GENERAL-ANSWER] System prompt length:', prompt.length);
    
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
