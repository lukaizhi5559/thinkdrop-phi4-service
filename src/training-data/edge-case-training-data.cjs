/**
 * Edge case training data
 * Includes unusual patterns, typos, mixed languages, and challenging examples
 */

const edgeCaseTrainingData = [
  // Typos and Misspellings
  { text: "Remeber I have a meting tomorow", intent: "memory_store" },
  { text: "Whn is my apointment?", intent: "memory_retrieve" },
  { text: "Tak a scrensht", intent: "command" },
  { text: "Wat is the captal of Frace?", intent: "question" },
  { text: "Helo ther", intent: "greeting" },

  // All Caps
  { text: "REMEMBER I HAVE A MEETING TOMORROW", intent: "memory_store" },
  { text: "WHEN IS MY APPOINTMENT?", intent: "memory_retrieve" },
  { text: "TAKE A SCREENSHOT", intent: "command" },
  { text: "WHAT IS THE CAPITAL OF FRANCE?", intent: "question" },
  { text: "HELLO", intent: "greeting" },

  // No Punctuation
  { text: "remember i have a meeting tomorrow at 3pm", intent: "memory_store" },
  { text: "when is my dentist appointment", intent: "memory_retrieve" },
  { text: "take a screenshot", intent: "command" },
  { text: "what is the capital of france", intent: "question" },
  { text: "hello there", intent: "greeting" },

  // Excessive Punctuation
  { text: "Remember!!! I have a meeting tomorrow!!!", intent: "memory_store" },
  { text: "When is my appointment???", intent: "memory_retrieve" },
  { text: "Take a screenshot!!!", intent: "command" },
  { text: "What is the capital of France??", intent: "question" },
  { text: "Hello!!!", intent: "greeting" },

  // Very Short
  { text: "remember meeting", intent: "memory_store" },
  { text: "when appointment", intent: "memory_retrieve" },
  { text: "screenshot", intent: "command" },
  { text: "capital france", intent: "question" },
  { text: "hi", intent: "greeting" },

  // Very Long
  { text: "I need you to remember that I have a very important meeting with my boss and several colleagues tomorrow afternoon at exactly 3:30pm in the main conference room on the 5th floor and I should bring my laptop, presentation slides, and quarterly report", intent: "memory_store" },
  { text: "Can you please tell me when exactly my dentist appointment is scheduled because I think it might be sometime next week but I'm not entirely sure and I need to plan my schedule accordingly", intent: "memory_retrieve" },
  { text: "Please take a screenshot of my entire screen right now and save it to my desktop folder with today's date in the filename", intent: "command" },
  { text: "What is the capital city of France and can you also tell me about its history, population, major landmarks, and cultural significance?", intent: "question" },
  { text: "Hello there! Good morning! How are you doing today? I hope you're having a wonderful day!", intent: "greeting" },

  // Numbers and Symbols
  { text: "Remember meeting @ 3pm w/ John & Sarah", intent: "memory_store" },
  { text: "When is appt #2?", intent: "memory_retrieve" },
  { text: "Screenshot -> desktop", intent: "command" },
  { text: "What is 2+2?", intent: "question" },
  { text: "Hi :)", intent: "greeting" },

  // Mixed Case
  { text: "ReMeMbEr I hAvE a MeEtInG", intent: "memory_store" },
  { text: "WhEn Is My ApPoInTmEnT?", intent: "memory_retrieve" },
  { text: "TaKe A sCrEeNsHoT", intent: "command" },
  { text: "WhAt Is ThE cApItAl Of FrAnCe?", intent: "question" },
  { text: "HeLLo", intent: "greeting" },

  // Emojis
  { text: "Remember 📅 meeting tomorrow 🕐 3pm", intent: "memory_store" },
  { text: "When is my appointment? 🤔", intent: "memory_retrieve" },
  { text: "Take a screenshot 📸", intent: "command" },
  { text: "What is the capital of France? 🇫🇷", intent: "question" },
  { text: "Hello! 👋😊", intent: "greeting" },

  // Partial Sentences
  { text: "remember that thing tomorrow", intent: "memory_store" },
  { text: "when is that", intent: "memory_retrieve" },
  { text: "do it", intent: "command" },
  { text: "what about", intent: "question" },
  { text: "hey", intent: "greeting" },

  // Double Negatives
  { text: "Don't not remember my meeting", intent: "memory_store" },
  { text: "I can't not forget when my appointment is", intent: "memory_retrieve" },
  { text: "Don't not take a screenshot", intent: "command" },
  { text: "What isn't not the capital of France?", intent: "question" },
  { text: "Not unhello", intent: "greeting" },

  // Questions as Statements
  { text: "I'm wondering when my appointment is", intent: "memory_retrieve" },
  { text: "I'd like to know the capital of France", intent: "question" },
  { text: "I'm curious about what we discussed", intent: "context" },
  { text: "I want to see my schedule", intent: "memory_retrieve" },
  { text: "I'm interested in taking a screenshot", intent: "command" },

  // Statements as Questions
  { text: "Tell me when my appointment is", intent: "memory_retrieve" },
  { text: "Explain the capital of France", intent: "question" },
  { text: "Show me what we talked about", intent: "context" },
  { text: "Give me my schedule", intent: "memory_retrieve" },
  { text: "Make a screenshot", intent: "command" },

  // Redundant Words
  { text: "Please remember to remember my meeting", intent: "memory_store" },
  { text: "When when is my appointment?", intent: "memory_retrieve" },
  { text: "Take take a screenshot", intent: "command" },
  { text: "What what is the capital?", intent: "question" },
  { text: "Hello hello", intent: "greeting" },

  // Foreign Words Mixed In
  { text: "Remember mi reunión tomorrow", intent: "memory_store" },
  { text: "Quand is my appointment?", intent: "memory_retrieve" },
  { text: "Take une screenshot", intent: "command" },
  { text: "What ist die capital of France?", intent: "question" },
  { text: "Bonjour hello", intent: "greeting" },

  // Incomplete Thoughts
  { text: "Remember I have a", intent: "memory_store" },
  { text: "When is my", intent: "memory_retrieve" },
  { text: "Take a", intent: "command" },
  { text: "What is the", intent: "question" },
  { text: "Hello I", intent: "greeting" },

  // Multiple Intents (should pick primary)
  { text: "Hello! Remember I have a meeting tomorrow", intent: "memory_store" },
  { text: "What is the capital of France? Also, when is my appointment?", intent: "question" },
  { text: "Take a screenshot and open Chrome", intent: "command" },
  { text: "Good morning! What did we discuss yesterday?", intent: "context" },
  { text: "Remember to buy milk. What's the weather like?", intent: "memory_store" },
];

module.exports = edgeCaseTrainingData;
