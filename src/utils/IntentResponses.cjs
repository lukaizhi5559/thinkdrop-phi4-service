/**
 * Suggested responses for different intent types
 */

const IntentResponses = {
  /**
   * Get suggested response for a given intent
   * @param {string} intent - Intent type
   * @param {string} message - Original user message
   * @param {Array} entities - Extracted entities
   * @returns {string} Suggested response
   */
  getSuggestedResponse(intent, message, entities = []) {
    switch (intent) {
      case 'memory_store':
        return this._getMemoryStoreResponse(message, entities);

      case 'memory_retrieve':
        return this._getMemoryRetrieveResponse(message, entities);

      case 'command':
        return this._getCommandResponse(message, entities);

      case 'question':
        return this._getQuestionResponse(message, entities);

      case 'greeting':
        return this._getGreetingResponse(message);

      case 'context':
        return this._getContextResponse(message);

      case 'farewell':
        return this._getFarewellResponse(message);

      default:
        return "I understand. How can I assist you?";
    }
  },

  _getMemoryStoreResponse(message, entities) {
    const personEntities = entities.filter((e) => e.type === 'person');
    const dateEntities = entities.filter((e) => e.type === 'datetime');

    if (personEntities.length > 0 && dateEntities.length > 0) {
      return `I'll remember that you have something with ${personEntities[0].value} ${dateEntities[0].value}.`;
    } else if (personEntities.length > 0) {
      return `I'll remember that about ${personEntities[0].value}.`;
    } else if (dateEntities.length > 0) {
      return `I'll remember that for ${dateEntities[0].value}.`;
    }

    return `I'll remember that: ${message.substring(0, 50)}${message.length > 50 ? '...' : ''}`;
  },

  _getMemoryRetrieveResponse(message, entities) {
    const personEntities = entities.filter((e) => e.type === 'person');
    const dateEntities = entities.filter((e) => e.type === 'datetime');

    if (personEntities.length > 0) {
      return `Let me search for information about ${personEntities[0].value}...`;
    } else if (dateEntities.length > 0) {
      return `Let me check what you have for ${dateEntities[0].value}...`;
    }

    return "Let me search for that information...";
  },

  _getCommandResponse(message, entities) {
    const lowerMessage = message.toLowerCase();

    if (lowerMessage.includes('screenshot')) {
      return "Taking a screenshot...";
    } else if (lowerMessage.includes('open')) {
      return "Opening the application...";
    } else if (lowerMessage.includes('close')) {
      return "Closing the application...";
    } else if (lowerMessage.includes('search')) {
      return "Searching...";
    }

    return "Executing command...";
  },

  _getQuestionResponse(message, entities) {
    const lowerMessage = message.toLowerCase();

    if (lowerMessage.startsWith('what')) {
      return "Let me explain that for you...";
    } else if (lowerMessage.startsWith('how')) {
      return "Here's how that works...";
    } else if (lowerMessage.startsWith('why')) {
      return "The reason is...";
    } else if (lowerMessage.startsWith('when')) {
      return "Let me check the timing...";
    } else if (lowerMessage.startsWith('where')) {
      return "Let me find that location...";
    } else if (lowerMessage.startsWith('who')) {
      return "Let me tell you about that...";
    }

    return "Let me answer that for you...";
  },

  _getGreetingResponse(message) {
    const lowerMessage = message.toLowerCase();

    if (lowerMessage.includes('morning')) {
      return "Good morning! How can I help you today?";
    } else if (lowerMessage.includes('afternoon')) {
      return "Good afternoon! What can I do for you?";
    } else if (lowerMessage.includes('evening')) {
      return "Good evening! How may I assist you?";
    }

    return "Hello! How can I help you today?";
  },

  _getContextResponse(message) {
    return "Let me review our conversation history...";
  },

  _getFarewellResponse(message) {
    return "Goodbye! Have a great day!";
  },
};

module.exports = IntentResponses;
