/**
 * Mathematical utility functions for NLP operations
 */

const MathUtils = {
  /**
   * Calculate cosine similarity between two vectors
   * @param {number[]} vecA - First vector
   * @param {number[]} vecB - Second vector
   * @returns {number} Cosine similarity score (0-1)
   */
  cosineSimilarity(vecA, vecB) {
    if (!vecA || !vecB || vecA.length !== vecB.length) {
      throw new Error('Vectors must have the same length');
    }

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < vecA.length; i++) {
      dotProduct += vecA[i] * vecB[i];
      normA += vecA[i] * vecA[i];
      normB += vecB[i] * vecB[i];
    }

    const denominator = Math.sqrt(normA) * Math.sqrt(normB);
    if (denominator === 0) return 0;

    return dotProduct / denominator;
  },

  /**
   * Calculate Euclidean distance between two vectors
   * @param {number[]} vecA - First vector
   * @param {number[]} vecB - Second vector
   * @returns {number} Euclidean distance
   */
  euclideanDistance(vecA, vecB) {
    if (!vecA || !vecB || vecA.length !== vecB.length) {
      throw new Error('Vectors must have the same length');
    }

    let sum = 0;
    for (let i = 0; i < vecA.length; i++) {
      const diff = vecA[i] - vecB[i];
      sum += diff * diff;
    }

    return Math.sqrt(sum);
  },

  /**
   * Normalize a vector to unit length
   * @param {number[]} vec - Vector to normalize
   * @returns {number[]} Normalized vector
   */
  normalize(vec) {
    const norm = Math.sqrt(vec.reduce((sum, val) => sum + val * val, 0));
    if (norm === 0) return vec;
    return vec.map((val) => val / norm);
  },

  /**
   * Calculate mean of an array of numbers
   * @param {number[]} arr - Array of numbers
   * @returns {number} Mean value
   */
  mean(arr) {
    if (!arr || arr.length === 0) return 0;
    return arr.reduce((sum, val) => sum + val, 0) / arr.length;
  },

  /**
   * Calculate standard deviation
   * @param {number[]} arr - Array of numbers
   * @returns {number} Standard deviation
   */
  standardDeviation(arr) {
    if (!arr || arr.length === 0) return 0;
    const avg = this.mean(arr);
    const squareDiffs = arr.map((val) => Math.pow(val - avg, 2));
    return Math.sqrt(this.mean(squareDiffs));
  },

  /**
   * Softmax function for probability distribution
   * @param {number[]} arr - Array of numbers
   * @returns {number[]} Softmax probabilities
   */
  softmax(arr) {
    const maxVal = Math.max(...arr);
    const exps = arr.map((val) => Math.exp(val - maxVal));
    const sumExps = exps.reduce((sum, val) => sum + val, 0);
    return exps.map((val) => val / sumExps);
  },
};

module.exports = MathUtils;
