/**
 * Knowledge-Based Answer Service
 * Uses Phi4 LLM's built-in knowledge as a fallback when web search fails
 * Provides factual answers with confidence estimation
 */

const llmService = require('./llm.js');

class KnowledgeService {
  constructor() {
    this.llmService = llmService;
  }

  /**
   * Answer a factual question using Phi4's built-in knowledge
   * @param {string} query - The question to answer
   * @param {object} options - Answer options
   * @returns {Promise<object>} Answer with confidence and disclaimer
   */
  async answerFromKnowledge(query, options = {}) {
    const {
      temperature = 0.1, // Low temperature for factual answers
      maxTokens = 300,
      minConfidence = 0.6
    } = options;

    console.log('🧠 [KNOWLEDGE] Answering from built-in knowledge...');
    console.log(`   Query: "${query}"`);

    // Construct prompt for factual answer
    const prompt = this.buildFactualPrompt(query);

    try {
      // Generate answer using Phi4
      const response = await this.llmService.generate(prompt, {
        temperature,
        max_tokens: maxTokens,
        stop: ['\n\nQuestion:', '\n\n---', 'Sources:']
      });

      if (!response || !response.text) {
        throw new Error('No response from LLM');
      }

      const answer = response.text.trim();

      // Estimate confidence based on response patterns
      const confidence = this.estimateConfidence(answer, query);

      console.log(`   Confidence: ${(confidence * 100).toFixed(1)}%`);

      // Check if confidence meets minimum threshold
      if (confidence < minConfidence) {
        console.log('   ⚠️  Confidence too low, admitting uncertainty');
        return {
          answer: "I don't have reliable information about that in my knowledge base. This query would benefit from a web search.",
          confidence: confidence,
          source: 'phi4_knowledge',
          reliable: false,
          disclaimer: 'Low confidence answer - web search recommended'
        };
      }

      return {
        answer: answer,
        confidence: confidence,
        source: 'phi4_knowledge',
        reliable: true,
        disclaimer: 'This answer is based on the AI model\'s training data (cutoff: April 2024) and may not reflect the most current information.'
      };

    } catch (error) {
      console.error('❌ [KNOWLEDGE] Error:', error.message);
      throw error;
    }
  }

  /**
   * Build a prompt optimized for factual answers
   * @param {string} query - The question
   * @returns {string} Formatted prompt
   */
  buildFactualPrompt(query) {
    return `You are a knowledgeable assistant answering a factual question. Provide a concise, accurate answer based on your training knowledge.

Rules:
1. Answer directly and concisely (2-3 sentences max)
2. State facts confidently if you know them
3. If uncertain, say "I'm not certain, but..." or "Based on my knowledge..."
4. Do not make up information
5. Do not provide sources or citations
6. Do not use phrases like "as of my last update" unless necessary

Question: ${query}

Answer:`;
  }

  /**
   * Estimate confidence in the answer
   * @param {string} answer - The generated answer
   * @param {string} _query - The original query (reserved for future use)
   * @returns {number} Confidence score (0-1)
   */
  estimateConfidence(answer, _query) {
    let confidence = 0.7; // Base confidence

    // Patterns indicating high confidence
    const highConfidencePatterns = [
      /^(yes|no),/i,
      /^the .+ is/i,
      /^it is/i,
      /^\d+/,
      /capital.*is/i,
      /located in/i,
      /known as/i,
      /defined as/i
    ];

    // Patterns indicating low confidence
    const lowConfidencePatterns = [
      /i don't know/i,
      /i'm not sure/i,
      /i cannot/i,
      /i don't have/i,
      /uncertain/i,
      /unclear/i,
      /it's difficult to say/i,
      /hard to determine/i,
      /may vary/i,
      /depends on/i,
      /could be/i,
      /might be/i,
      /possibly/i,
      /perhaps/i
    ];

    // Patterns indicating hedging (medium-low confidence)
    const hedgingPatterns = [
      /based on my knowledge/i,
      /as far as i know/i,
      /generally/i,
      /typically/i,
      /usually/i,
      /often/i,
      /in most cases/i
    ];

    // Check for high confidence patterns
    for (const pattern of highConfidencePatterns) {
      if (pattern.test(answer)) {
        confidence += 0.15;
        break;
      }
    }

    // Check for low confidence patterns
    for (const pattern of lowConfidencePatterns) {
      if (pattern.test(answer)) {
        confidence -= 0.4;
        break;
      }
    }

    // Check for hedging patterns
    for (const pattern of hedgingPatterns) {
      if (pattern.test(answer)) {
        confidence -= 0.15;
        break;
      }
    }

    // Length-based adjustment (very short or very long answers may be less confident)
    const wordCount = answer.split(/\s+/).length;
    if (wordCount < 5) {
      confidence -= 0.1; // Too short
    } else if (wordCount > 100) {
      confidence -= 0.1; // Too verbose (may be uncertain)
    }

    // Question mark in answer (indicates uncertainty)
    if (answer.includes('?')) {
      confidence -= 0.15;
    }

    // Clamp confidence between 0 and 1
    return Math.max(0, Math.min(1, confidence));
  }

  /**
   * Check if a query is suitable for knowledge-based answering
   * @param {string} query - The query to check
   * @returns {boolean} True if suitable
   */
  isSuitableForKnowledge(query) {
    // Queries that are NOT suitable for knowledge-based answering
    const unsuitablePatterns = [
      /current|latest|recent|today|now|this (week|month|year)/i,
      /price|cost|stock|market/i,
      /weather|forecast/i,
      /news|breaking/i,
      /\d{4}/ // Contains a year (likely asking about recent events)
    ];

    for (const pattern of unsuitablePatterns) {
      if (pattern.test(query)) {
        return false;
      }
    }

    return true;
  }
}

module.exports = KnowledgeService;
