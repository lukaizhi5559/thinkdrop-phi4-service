/**
 * Enhanced training data with more complex examples
 * Includes multi-intent scenarios, context-dependent examples, and edge cases
 */

const enhancedTrainingData = [
  // Complex Memory Store
  { text: "Remember that I have a doctor's appointment with Dr. Johnson at the downtown clinic on Thursday at 2:30pm and I need to bring my insurance card", intent: "memory_store" },
  { text: "Note: team meeting every Monday and Wednesday at 10am in conference room B with Sarah, Mike, and Lisa", intent: "memory_store" },
  { text: "Save this for later: my car registration expires on March 15th, 2025 and I need to renew it at the DMV", intent: "memory_store" },
  { text: "Remember I promised to help John move to his new apartment at 123 Oak Street on Saturday morning", intent: "memory_store" },
  { text: "Keep track: I borrowed $200 from David last week and need to pay him back by Friday", intent: "memory_store" },

  // Complex Memory Retrieve
  { text: "What appointments do I have this week and who are they with?", intent: "memory_retrieve" },
  { text: "When and where is my next team meeting?", intent: "memory_retrieve" },
  { text: "What do I need to renew soon and when does it expire?", intent: "memory_retrieve" },
  { text: "Who did I promise to help and when?", intent: "memory_retrieve" },
  { text: "How much money do I owe and to whom?", intent: "memory_retrieve" },

  // Complex Commands
  { text: "Take a screenshot and save it to my desktop", intent: "command" },
  { text: "Open Chrome and navigate to Gmail", intent: "command" },
  { text: "Search for Italian restaurants within 5 miles", intent: "command" },
  { text: "Set a timer for 25 minutes and start playing focus music", intent: "command" },
  { text: "Open my calendar and show me next week's schedule", intent: "command" },

  // Complex Questions
  { text: "What is the difference between machine learning and deep learning?", intent: "question" },
  { text: "How does climate change affect ocean temperatures and marine life?", intent: "question" },
  { text: "What are the main causes of the French Revolution and when did it occur?", intent: "question" },
  { text: "Can you explain the theory of relativity in simple terms?", intent: "question" },
  { text: "What is the relationship between supply and demand in economics?", intent: "question" },

  // Conversational Greetings
  { text: "Hey there! How's your day going?", intent: "greeting" },
  { text: "Good morning! Ready to help me out today?", intent: "greeting" },
  { text: "Hi! I hope you're doing well", intent: "greeting" },
  { text: "Hello assistant, nice to chat with you again", intent: "greeting" },
  { text: "What's up? Long time no see!", intent: "greeting" },

  // Context-Heavy Examples
  { text: "What were the details of that appointment we discussed?", intent: "context" },
  { text: "Can you remind me what you said about the meeting?", intent: "context" },
  { text: "Go back to when we were talking about my schedule", intent: "context" },
  { text: "What was that restaurant name you mentioned earlier?", intent: "context" },
  { text: "Repeat the instructions you gave me before", intent: "context" },

  // Ambiguous Examples (require context)
  { text: "Tell me more about that", intent: "context" },
  { text: "When is it?", intent: "memory_retrieve" },
  { text: "Who's coming?", intent: "memory_retrieve" },
  { text: "Do it now", intent: "command" },
  { text: "Show me", intent: "command" },

  // Multi-clause Examples
  { text: "I have a meeting tomorrow but I can't remember what time it is", intent: "memory_retrieve" },
  { text: "Remember to buy groceries and also pick up the dry cleaning", intent: "memory_store" },
  { text: "What is photosynthesis and how does it relate to climate change?", intent: "question" },
  { text: "Open my email and check if there are any messages from Sarah", intent: "command" },
  { text: "Good morning! Can you remind me what I need to do today?", intent: "memory_retrieve" },

  // Negation Examples
  { text: "Don't forget I have a meeting at 3pm", intent: "memory_store" },
  { text: "I don't remember when my appointment is", intent: "memory_retrieve" },
  { text: "Don't open that file", intent: "command" },
  { text: "I can't recall what we discussed", intent: "context" },
  { text: "Never mind, I found it", intent: "greeting" },

  // Polite/Formal Examples
  { text: "Could you please remember that I have a meeting tomorrow?", intent: "memory_store" },
  { text: "Would you mind telling me when my appointment is?", intent: "memory_retrieve" },
  { text: "I would appreciate it if you could take a screenshot", intent: "command" },
  { text: "May I ask what the capital of France is?", intent: "question" },
  { text: "Good evening, I hope I'm not bothering you", intent: "greeting" },

  // Casual/Slang Examples
  { text: "Yo, remember I gotta meet John tomorrow", intent: "memory_store" },
  { text: "When's that thing I gotta do?", intent: "memory_retrieve" },
  { text: "Snap a pic of this", intent: "command" },
  { text: "What's the deal with quantum physics?", intent: "question" },
  { text: "Sup dude", intent: "greeting" },

  // Time-Sensitive Examples
  { text: "Remember I have a meeting in 30 minutes", intent: "memory_store" },
  { text: "What do I have scheduled for right now?", intent: "memory_retrieve" },
  { text: "Set an alarm for 5 minutes from now", intent: "command" },
  { text: "What time is it in Tokyo right now?", intent: "question" },
  { text: "Good morning! It's 6am already!", intent: "greeting" },

  // Location-Based Examples
  { text: "Remember I parked in lot C, space 42", intent: "memory_store" },
  { text: "Where did I say I was meeting Sarah?", intent: "memory_retrieve" },
  { text: "Find coffee shops near me", intent: "command" },
  { text: "What is the population of New York City?", intent: "question" },
  { text: "Hello from San Francisco!", intent: "greeting" },
];

module.exports = enhancedTrainingData;
