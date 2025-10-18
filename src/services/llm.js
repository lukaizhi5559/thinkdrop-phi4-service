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

    // If LLM is disabled, return placeholder
    if (!this.enabled) {
      return this._generatePlaceholderResponse(query, options, startTime);
    }

    try {
      const prompt = this._buildPrompt(query, options.context);
      
      const response = await axios.post(this.apiUrl, {
        model: options.model || this.model,
        prompt: prompt,
        stream: false,
        options: {
          temperature: options.temperature || 0.3,  // Lower for more focused answers
          num_predict: options.maxTokens || 100,    // Much shorter default
          top_k: options.topK || 40,
          top_p: options.topP || 0.9,
          num_ctx: options.contextLength || 2048,
          stop: ['\n\nUser:', '\n\n###', '<|end|>', 'Instruction']
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

  _buildPrompt(query, context) {
    // Build a concise prompt with system instructions
    let prompt = 'You are a helpful assistant. Provide concise, direct answers without extra explanations unless asked.\n\n';
    
    if (context && context.length > 0) {
      const contextStr = context.map(c => `${c.role}: ${c.content}`).join('\n');
      prompt += `${contextStr}\n\n`;
    }
    
    prompt += `User: ${query}\n\nAssistant:`;
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
