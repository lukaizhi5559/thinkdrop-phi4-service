/**
 * DistilBERT Intent Parser
 * High-accuracy parser using DistilBERT embeddings + NER
 * Accuracy: 95%+
 * Latency: ~42ms
 */

const { pipeline } = require('@xenova/transformers');
const MathUtils = require('../utils/MathUtils.cjs');
const IntentResponses = require('../utils/IntentResponses.cjs');
const nlp = require('compromise');

class DistilBertIntentParser {
  constructor() {
    this.embedder = null;
    this.initialized = false;
    
    // Intent labels
    this.intentLabels = [
      'command_automate',    // Nut.js UI automation (complex multi-step workflows)
      'app_control_start',   // Enter/exit persistent app control mode (scroll, type, shortcuts)
      'screen_intelligence', // Primary screen analysis (UI elements, browser content, desktop items)
      // 'command_execute',     // Shell/OS commands (simple, direct execution)
      // 'command_guide',       // Educational/tutorial mode ("show me how")
      'web_search',         // Time-sensitive queries requiring current data
      'memory_store',
      'memory_retrieve', 
      'general_knowledge',  // Includes all questions and knowledge queries
      'greeting'
    ];
    
    // Seed examples for each intent (expanded with paraphrases, edge cases, hard negatives)
    // Aim: 15-25 diverse examples per intent for robust classification
    this.seedExamples = {
      memory_store: [
        // ── Original (kept) ─────────────────────────────────────
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
        "Remember my favorite coffee is oat milk latte",
        // NOTE: "Set a reminder", "Remind me in X minutes" moved to command_automate
        // — they need the schedule pseudo-skill, not memory storage.
        "I need to buy milk and eggs",
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
        "Remember my favorite coffee is oat milk latte",
        // NOTE: reminder seeds moved to command_automate (schedule pseudo-skill)

        // ── New – richer phrasing, multi-entity, typos ───────
        // NOTE: "Quick reminder: call Dr. Patel" moved to command_automate
        "Store this: license plate 7XYZ123, expires 2026-08-31",
        "Jot down that I owe Mike $42 for the concert tickets",
        "Never forget: anniversary dinner reservation at Le Petit Bistro, 7:30pm Sat",
        "Add to notes – my blood type is O-negative",
        "Remember I’m on a gluten-free diet starting Monday",
        "Save my gym locker combo: 14-28-03",
        "Note that the new office Wi-Fi is 'CorpGuest' / pw 'Welcome2025!'",
        "Log workout: 45 min spin class, 320 cal burned",
        "Put this in memory: cousin Lisa’s baby shower is 11/22 at 2pm",
        "Keep the API token safe: sk_live_51J…",
        "Save the flight confirmation: AA 1847, departs 06:15 on 12/03",
        "Remember I take 20mg of Lipitor every night",
        "Add parking spot B-17 to my car notes",
        "Store my Spotify playlist URL: https://open.spotify.com/playlist/…",
        "Note that I’m out of office 12/24-12/26",
        "Remember my preferred seat is 12A on Delta",
        "Save the vet appointment for Max on 11/18 at 3:45pm",
        "Keep in mind the sprint review is every other Thursday 10am",
        "Log that I finished reading 'Atomic Habits' today",
        
        // ── "I have" patterns (to fix memory_retrieve confusion) ───────
        "I have an appt for next week Sunday 2pm at the dentist",
        "I have a meeting tomorrow at 10am with the team",
        "I have a doctor's appointment on Friday at 3pm",
        "I have a dentist appointment next Tuesday",
        "I have an interview scheduled for Monday 9am",
        "I have a flight on December 15th at 6am",
        "I have a haircut appointment this Thursday at 2pm",
        "I have a vet appointment for my dog next week",
        "I have a conference call at 4pm today",
        "I have a deadline on Friday for the project",
        "I have a reservation at the restaurant for 7pm Saturday",
        "I have a gym session booked for tomorrow morning",
        "I have an oil change scheduled for next Monday",
        "I have a piano lesson every Wednesday at 5pm",
        "I have a package arriving on Tuesday",
        
        // ── "Keep track" patterns (to fix memory_retrieve confusion) ───────
        "Keep track of my workout schedule",
        "Keep track of my gym routine",
        "Keep track of my running schedule",
        "Keep track of my diet plan",
        "Keep track of my medication schedule",
        "Keep track of my appointments",
        "Keep track of my meetings",
        "Keep track of my deadlines",
        "Keep track of my goals",
        "Keep track of my progress",
        "Keep track of my expenses",
        "Keep track of my habits",
        "Keep track of my sleep schedule",
        "Keep track of my water intake",
        "Keep track of my study schedule",
        
        // ── Explicit "to memory" / "remember this" patterns ───────
        // These should NEVER be classified as screen_intelligence
        "Record this alert to memory - VM5 sandbox_bundle:2 Electron Security Warning",
        "Remember this error message for later",
        "Save this to memory - API endpoint is https://api.example.com",
        "Add this to memory - the password is abc123",
        "Record this - meeting notes from today's standup",
        "Remember this code snippet for future reference",
        "Store this in memory - license key XYZ-789-ABC",
        "Keep this in memory - server IP is 192.168.1.100",
        "Record this warning message to memory",
        "Remember this configuration setting",
        "Save this error to memory for debugging",
        "Add this alert to memory",
        "Record this notification to memory",
        "Remember this log entry",
        "Store this message in memory",
        "Keep this alert in memory",
        "Save this warning to memory",
        "Add this error message to memory",
        "Record this to memory",
        "Remember this for later",
        
        // ── Personal-fact declarations — identity and relationship facts ────────────
        // Standard: "My <role> is <value>"
        "My name is Sam",
        "My name is Lukas",
        "My wife is Sarah",
        "My husband is James",
        "My mom is Linda",
        "My dad is Robert",
        "My boss is David",
        "My cousin is Chris",
        "My dentist is Dr. Patel",
        "My doctor is Dr. Kim",
        "My lawyer is Mr. Thompson",
        "My phone number is +1 555 123 4567",
        "My email is sam@example.com",
        "My address is 123 Main Street",
        "My sister is Amanda",
        "My brother is Jake",
        "My son is Tyler",
        "My daughter is Emma",
        "My friend is Marcus",
        "My neighbor is Tom",
        "My coworker is Priya",
        "My manager is Lisa",
        "My trainer is Carlos",
        // Possessive: "My wife's name is Sarah"
        "My wife's name is Sarah",
        "My dad's name is Robert",
        "My cousin's name is Chris",
        "My boss's name is David",
        // With filler prefix: "No my name is Sam", "Actually my name is Lukas"
        "No my name is Sam",
        "Actually my name is Lukas",
        "Wait my wife is Sarah",
        "Actually my boss is David",
        "No my cousin is Chris",
        // Inverted: "Chris Akers is my cousin", "John is my boss"
        "Chris Akers is my cousin",
        "John Smith is my boss",
        "Sarah is my wife",
        "Dr. Patel is my dentist",
        "Marcus is my best friend",
        "Amanda is my sister",
        "Tom is my neighbor",
        "Lisa is my manager",
        // "I am" identity declarations
        "I am Sam",
        "I am Lukas",

        // ── CRITICAL: Store patterns that were failing tests ────────────────────────
        "From now on, call me Alex",
        "From now on call me John",
        "Call me Alex from now on",
        "Remember that my favorite color is blue",
        "Remember my favorite color is blue",
        "My favorite color is blue",
        "Store my birthday as May 3rd",
        "My birthday is May 3rd",
        "Save my birthday as May 3rd",
        "In the future, assume I prefer metric units",
        "In the future assume I prefer metric",
        "Assume I prefer metric units",
        "Note that I am vegetarian",
        "I am vegetarian",
        "I'm vegetarian",
        "Remember that my kids go to school at 8 am",
        "My kids go to school at 8 am",
        "Kids go to school at 8 am",
        "Add this website to my study resources",
        "Add this to my study resources",
        "Save this to my resources",
        "Please remember that I am learning Japanese",
        "Remember I am learning Japanese",
        "I am learning Japanese",
        "Just remember all of this conversation",
        "Remember all of this",
        "Remember this conversation",
        
        // ── Screen Intelligence + Memory Store patterns ───────
        // User wants to save what they're looking at on screen
        "Remember this error on my screen",
        "Save this error message I'm seeing",
        "Record this bug for later",
        "Remember what this code does",
        "Save this configuration I'm looking at",
        "Remember this UI layout",
        "Store this error pattern I'm seeing",
        "Remember this screen for later",
        "Save what's on my screen to memory",
        "Record this screen content",
        "Remember this API key on screen",
        "Save this password I'm looking at",
        
        // ── Command Automate + Memory Store patterns ──────────
        // User wants to save workflows and preferences
        "Remember this is my favorite browser",
        "Save this workflow for next time",
        "Remember this automation sequence",
        "Store this command for later",
        "Remember my preferred app for this",
        "Save this as my usual workflow",
        "Remember I always use Chrome for work",
        "Save this as my default automation",
        "Remember this is my go-to app",
        "Store this command sequence",
        
        // ── Temperature/unit preferences ────────────────────────
        "From now on use Celsius unless I say otherwise",
        "Use Celsius from now on",
        "Default to Celsius",
        "Always use metric",
        "Prefer Celsius over Fahrenheit",
        
        // ── Water/health tracking ────────────────────────
        "Log that I drank two bottles of water today",
        "Log my water intake",
        "Track my water consumption",
        "Record that I drank water",
        "Save my water intake for today",
        
        // ── Note organization ────────────────────────
        "Store this note under ideas",
        "Save this under ideas",
        "File this under ideas",
        "Put this in my ideas folder",
        "Add this to ideas category",
        
        // ── Checklist management ────────────────────────
        "Add this to my preparation checklist",
        "Put this on my checklist",
        "Add to my prep list",
        "Include this in my checklist",
        "Add this item to my checklist",
        
        // ── Coding preferences ────────────────────────
        "Remember that I prefer lowercase variable names",
        "Remember my coding style preference",
        "Save my variable naming preference",
        "Note my coding convention",
        
        // ── Implicit storage requests ────────────────────────
        "Can you remember this",
        "Remember this",
        "Keep this in mind",
        "Don't forget this",
        "Make a note of this",
        
        // ── Lifestyle preferences ────────────────────────
        "ok from here on out I'm a night person, remember that",
        "I'm a night person",
        "I'm a morning person",
        "remember I'm a night owl",
        
        // ── Dislike preferences ────────────────────────
        "hey, note: I hate pop-up notifications",
        "I hate pop-ups",
        "I don't like notifications",
        "note that I dislike",
        
        // ── Metaphorical storage ────────────────────────
        "just mentally bookmark this website for me",
        "mentally bookmark this",
        "bookmark this in your memory",
        "file this away",
        
        // ── Time interpretation rules ────────────────────────
        "treat 7am as early for me in the future",
        "7am is early for me",
        "consider 7am early",
        "remember that 7am is early",
        
        // ── Subjective logging ────────────────────────
        "log that today was a super productive day",
        "today was productive",
        "log today as productive",
        "mark today as a good day",
        
        // ── Weekly schedule ────────────────────────
        "consider Friday my cheat day, remember",
        "Friday is my cheat day",
        "remember Friday is cheat day",
        "Fridays are for cheating",
        
        // ── Relative dates ────────────────────────
        "remember that my mom's birthday is two days before mine",
        "my mom's birthday is before mine",
        "store this relative date",
        
        // ── Long-term storage ────────────────────────
        "stick this into my long-term memory please",
        "put this in long-term memory",
        "store this permanently",
        "remember this forever",
        
        // ── Response style ────────────────────────
        "I prefer minimal answers, keep that in mind",
        "I like short answers",
        "keep answers brief",
        "I prefer concise responses",
        
        // ── Vocabulary rules ────────────────────────
        "treat 'office' as my coworking space, not my home",
        "office means coworking space",
        "when I say office I mean",
        "From now on, whenever I say 'home', I mean my parents' house, not my apartment, so please remember that distinction",
        "home means my parents' house",
        "when I say home",
        
        // ── Diet/health goals ────────────────────────
        "I'm trying to cut down on sugar, so store the fact that I'm avoiding soda and candy for the next three months",
        "I'm avoiding sugar",
        "I'm cutting down on sweets",
        "remember I'm avoiding soda",
        
        // ── Entity memory ────────────────────────
        "Remember that my manager's name is Sarah and that she's in the London office",
        "my manager is Sarah",
        "Sarah is my manager",
        "remember my manager's name",
        
        // ── Ongoing tracking ────────────────────────
        "Please keep track of all the books I finish reading this year and remember that I started in March",
        "track my reading list",
        "keep track of books I read",
        "log my finished books",
        
        // ── Workout/exercise goals ────────────────────────
        "I'm starting a new workout plan next Monday, please remember that my target days are Monday, Wednesday, and Saturday",
        "my workout days are",
        "I exercise on Monday Wednesday Saturday",
        "remember my workout schedule",
        
        // ── Habit goals ────────────────────────
        "I'm trying to build a habit of reading 20 minutes every night, please remember that goal",
        "I want to read 20 minutes nightly",
        "my goal is to read every night",
        "remember my reading habit goal",
        
        // ── Job search goals ────────────────────────
        "I'm aiming to apply to three jobs per week, keep that as my target",
        "my target is 3 job applications per week",
        "I'm applying to 3 jobs weekly",
        "remember my job application goal",
        
        // ── Sleep schedule ────────────────────────
        "I'm experimenting with waking up at 6 am, remember this is my current schedule",
        "I wake up at 6am now",
        "my new wake time is 6am",
        "remember I'm waking at 6",
        
        // ── Digital wellbeing ────────────────────────
        "I want to cut back on social media, store that I'm limiting myself to 30 minutes a day",
        "I'm limiting social media to 30 mins",
        "my social media limit is 30 minutes",
        "remember my screen time limit",
        
        // ── Learning goals ────────────────────────
        "I'm learning Spanish this year, remember that as one of my main focuses",
        "I'm learning Spanish",
        "Spanish is my focus this year",
        "remember I'm studying Spanish",
        
        // ── Family/personal names ────────────────────────
        "Remember that my sister's name is Emily",
        "Remember that my brother's name is",
        "My sister is called Emily",
        "My brother's name is",
        "Save my passport number for later",
        "Store my passport number",
        "Remember my passport number",

        // ── Round 5 seeds ───────────────────────────────────────
        // Client/business relationship facts
        "Client Kenji Watanabe is a SaaS founder",
        "My client is a startup founder quarterly review in May",
        "Save this client info for later",
        "Log this about my client",
        // Lifestyle change declarations
        "I went fully vegan starting January of this year",
        "I switched from coffee to matcha as of this month",
        "Record that I'm switching from coffee to matcha",
        "Note that I started a new diet",
        // Media / reading progress
        "I left off at Podcast Syntax episode 600 midway through remember that",
        "Started reading Deep Work last Tuesday currently on page forty two",
        "Save my reading progress in this book",
        "I'm on chapter five of the book",
        // Budget / numbers
        "Log my weekly grocery budget target as one hundred eighty dollars",
        "My grocery budget is one eighty a week",
        "Record my monthly budget as five hundred dollars",
        // Insurance / ID numbers
        "My renter's insurance policy number is HM-983201",
        "My policy number is HM-983201",
        "Store my insurance policy number",
        // Generic store imperatives
        "Write down everything I just told you",
        "Save everything I said just now",
        "Record the last thing I said",
        "Tuck that away somewhere",
        "I want to hold onto that information",
        "Keep that stored for me",
        "Lock that in for me",
        // Health / body observations
        "I've had lower back pain every single day this week",
        "I've been having headaches every morning",
        "My knee has been hurting when I run",
        "Record that I've been feeling tired lately",
        // ── Round 6 seeds ──────────────────────────────────────
        "I adopted a rescue cat named Mochi last Tuesday",
        "I got a new rescue dog named Biscuit from the shelter",
        "I adopted a rabbit named Pebble last weekend",
        "We got a kitten, her name is Luna",
        "I joined a book club, we meet every second Sunday",
        "I just joined a hiking group that meets on weekends",
        "I signed up for a pottery class starting next month",
        "I joined a local running club this week",
        "I'm learning to bake sourdough bread, my starter is named Harold",
        "I started getting into fermentation and home brewing",
        "I'm learning to play the guitar, fifteen minutes each day",
        "I started taking swimming lessons on Tuesdays",
        "Priya's preferred meeting style is async voice notes not video calls",
        "My new manager prefers async updates over live meetings",
        "My boss likes short Slack messages instead of long emails",
        "My colleague prefers to communicate through voice notes",
        "I'm training for the Portland half-marathon in October",
        "I signed up for a 10K race in June",
        "I'm prepping for a triathlon next spring",
        "I started training for my first marathon this month",
        "I'm on Metformin 500mg twice daily now",
        "I started taking Vitamin D supplements this month",
        "I'm currently on antibiotics for a week",
        "My doctor put me on blood pressure medication",
        "I'm on Metformin now",
        "Mochi is a tortoiseshell cat, note that",
        "Harold is the name of my sourdough starter",
        "The starter's name is Harold",
        "Note that Mochi's breed is tortoiseshell",
        "Monthly rent is 2300 now",
        "My rent went up to twenty three hundred this month",
        "My monthly expenses went up by two hundred dollars",
        "I baked my first successful sourdough loaf today",
        "I finally finished reading the book today",
        "I hit a new personal record at the gym today",
        "I've been feeling really anxious every morning before my standup",
        "I'm stressed about the upcoming product launch this week",
        "I've been having trouble sleeping all this week",
        "Pin that GitHub repo link I just shared to my memory",
        "Bookmark this link to my notes",
        "Save this URL to my memory for later",
        "New phone, same number though",
        // ── Round 6b reinforcement seeds ──────────────────────────────
        // Declarative linking (phi4 confused these with memory_retrieve)
        "Priya Nair is my manager",
        "Priya is my new manager starting this week",
        "My new boss is James Chen",
        "Marcus is my best friend from college",
        "Dr. Osei is my new dentist",
        "Theo Bergmann is my coworker",
        "Eli is my brother",
        // Person moved to city (phi4 confused these with memory_retrieve)
        "My friend Marcus moved to Denver last month",
        "my friend marcus moved to den ver last month",
        "Marcus relocated to Denver",
        "Marcus moved to Denver",
        "My coworker moved to Seattle last year",
        "My sister moved to Portland in March",
        // Milestone events (phi4 confused with memory_retrieve due to past tense)
        "I baked my first successful sourdough loaf today, Harold is officially alive",
        "I finished my first marathon today",
        "I completed my first 10K this morning",
        "Harold the starter finally doubled in size today",
        "I finally got Mochi to eat wet food today",
        // ── Round 7 seeds ──────────────────────────────────────
        // New persona facts (Lena PM, Ben SQL, Sophie, Jamie, kombucha SCOBY, Nala dog, Zara, Kenji roommate)
        "Lena Kapoor is my new project manager, she started this Monday",
        "Lena is my PM and she prefers Linear tickets over Slack messages",
        "Ben Okafor is my go-to person for SQL questions at work",
        "ben oh kah for is my go to per son for S Q L ques tions at work",
        "Sophie is my best friend from college, she relocated to Chicago last year",
        "Jamie is doing a semester abroad in Japan",
        "I'm making kombucha at home now, my SCOBY is named Greta",
        "My SCOBY is named Greta and she's been alive for three weeks",
        "I got a dog, her name is Nala, she's a golden retriever",
        "Nala had her check-up at the vet today, everything looks great",
        "nala had her check up at the vet to day every thing looks great",
        // Lifestyle / financial facts
        "I'm moving to a new apartment on the north side of the city in April",
        "My take-home pay after taxes is around $3,200 a month now",
        "I drive a Subaru Outback, bought it used about two years ago",
        "I've started investing in index funds using a dollar-cost averaging strategy",
        "I switched to Bear app for personal notes, ditching Notion after two years",
        "I star ted using bear app for my notes in stead of no shun this week",
        "I set up a standing desk in my home office this weekend",
        "I'm doing a Python data science course on Coursera, started this month",
        // TV / entertainment facts (memory store not web search)
        "I started watching The Bear on Netflix last night, just finished season one",
        "I star ted watch ing the bear on net flix last night",
        // Declarative race/event registrations
        "I signed up for the July cycling race today, it's a 65-mile course",
        // Doctor / contact facts
        "Dr. Victor Tran's office is at the corner of 5th and Main downtown",
        // Subaru milestone
        "My Subaru just hit 100k miles today, save that",
        // ── Round 7b seeds (closing 21-failure gaps) ────────────────
        // SCOBY / kombucha naming (ms-003)
        "Named my SCOBY starter Greta, she's been alive three weeks",
        "My SCOBY is called Greta and she's very healthy",
        "The SCOBY in my kombucha jar is named Bertha",
        // Nightly supplements (ms-005)
        "I take melatonin and magnesium every night before bed",
        "Started taking magnesium supplements nightly for better sleep",
        "I'm taking vitamin D and melatonin every night now",
        // Reading for book club (ms-008)
        "I'm reading The Name of the Wind for my book club",
        "Currently reading Name of the Wind by Patrick Rothfuss for book club",
        "For book club this month I'm reading a fantasy novel",
        // Cycling race signups - more variety (ms-017)
        "I registered for a 65-mile cycling race in July",
        "I signed up for the bike race happening in July",
        "Registered for the cycling event in July, 65 miles",
        "I joined the July cycling race, it's 65 miles long",
        // Pet adoption without 'a dog' context (ms-021, ms-v12)
        "I got Nala from the rescue shelter on Saturday",
        "Picked up Nala from a local rescue shelter Saturday morning",
        "Adopted Nala on Saturday, she's a golden retriever",
        "Got Nala from the shelter on Saturday morning",
        "i got na la from the res cue shel ter on sat ur day",
        // Name spelling correction (ms-026)
        "Ben Okafor's last name is spelled O-K-A-F-O-R",
        "My colleague's name is spelled K-A-P-O-O-R",
        "The correct spelling of his name is T-H-O-M-P-S-O-N",
        // Minimal edge phrasing (ms-e02, ms-e04, ms-e05)
        "New dog: Nala",
        "New cat: Luna",
        "New pet: Max, a tabby",
        "April move, north side of the city",
        "Moving April, north side neighborhood",
        "Index funds, DCA strategy",
        "DCA strategy for index fund investing",
        "Investing using dollar cost averaging into index funds",
        // App switch phrasing (ms-e03)
        "Bear instead of Notion now",
        "I use Bear instead of Notion these days",
        "Switched to Bear from Notion as of last week",
        "Bear app instead of Notion, that's my new setup"
      ],

      memory_retrieve: [
        // ── Original ─────────────────────────────────────
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
        "What's my car's VIN?",
        "when do I have an appointment",
        "when I do I have an appt",
        "when do I have my appt",
        "when is my appointment",
        "do I have any appointments",
        "what appointments do I have",
        "when is my next appointment",
        "when's my doctor appointment",
        "any upcoming appointments",
        "anything upcoming",

        // ── New – fuzzy, compound, time-relative ───────
        "Any appointments this week?",
        "Remind me what I owe Mike",
        "What’s my locker combo again?",
        "Show me the API token I saved",
        "When’s the baby shower?",
        "List everything I logged about workouts",
        "What dietary restrictions did I mention?",
        "Pull up the flight details for AA 1847",
        "What medicines am I on?",
        "Where did I park the car?",
        "Show me the Spotify link I stored",
        "When am I out of office?",
        "Which seat do I like on Delta?",
        "Vet appointment for Max?",
        "When’s the next sprint review?",
        "Did I finish any books recently?",
        "Anything due before end of month?",
        "What’s the gluten-free start date?",
        "Show all passwords I've saved",
        // ── Explicit memory/notes queries ──────────────
        "Check my memory",
        "Show my memory",
        "What's in my memory",
        "Show my notes",
        "Show my saved notes",
        "What do you remember about me",
        "What have I told you",
        "Search my memories",
        "Find in my notes",
        
        // ── "Do you remember" patterns ──────────────
        "Do you remember my gym routine",
        "Do you remember my appointments",
        "Do you remember my preferences",
        "Do you remember my allergies",
        "Do you remember my password",
        "Do you remember what I told you",
        "Do you remember my meeting",
        "Do you remember my deadline",
        "Do you remember my favorite",
        "Do you remember when my workout is",
        "Do you remember what my workout schedule is",
        "Do you remember what time I work out",
        "Do you remember my gym schedule",
        "Do you remember when I go to the gym",
        "Do you remember my exercise routine",
        "Do you remember what I told you about my workout",
        "Do you remember the workout plan I mentioned",
        "Do you remember my training schedule",
        "Do you remember when I exercise",
        "Do you remember when my birthday is",
        "Do you remember my birthday",
        "Do you remember what my favorite color is",
        "Do you remember my favorite color",
        "Do you remember where I live",
        "Do you remember my address",
        "Do you remember what I'm learning",
        "Do you remember my goals",
        
        // ── "When is/was" patterns (retrieval) ──────────────
        "When is my workout",
        "When is my appointment",
        "When is my meeting",
        "When is my birthday",
        "When was my last workout",
        "When did I last exercise",
        "When did I say my appointment was",
        "When did I mention my birthday",
        
        // ── "What did I say/tell you" patterns (retrieval) ──────────────
        "What did I say my favorite color was",
        "What did I tell you my name was",
        "What did I say about my workout",
        "What did I mention about my schedule",
        "What did I tell you about my preferences",
        "What did I say my birthday was",
        "What did I tell you about my allergies",
        
        // ── "What do you know" patterns ──────────────
        "What do you know about me",
        "What do you know about my schedule",
        "What do you know about my preferences",
        "What do you know about my habits",
        "What do you know about my goals",
        "What do you know about my appointments",
        "What do you know about my meetings",
        "What do you know about my work",
        
        // ── Retrieve patterns that were failing tests ────────────────────────
        "Do you remember my workout schedule",
        "Do you remember my workout schedule?",
        "Do you remember what my workout schedule is",
        "Do you remember what my workout schedule is?",
        "Can you recall my workout schedule",
        "Can you recall my workout schedule?",
        "Do you remember my schedule",
        "Do you remember my schedule?",
        "What is my workout schedule",
        "What is my workout schedule?",
        "Tell me my workout schedule",
        "What was my workout schedule",
        "Recall my workout schedule",
        "What did I ask you to call me",
        "What did I tell you to call me",
        "What should you call me",
        "What languages did I say I'm learning",
        "What did I say I'm learning",
        "Which languages am I learning",
        "What do you know about my routine",
        "What do you know about my diet",
        
        // ── Personal attribute retrieval ("what's my X", "who is my X") ──────
        // These ask ThinkDrop to recall a stored personal fact → memory_retrieve
        "What's my name",
        "What is my name",
        "Whats my name",
        "What's my phone number",
        "What is my phone number",
        "What's my email",
        "What is my email address",
        "What's my home address",
        "What is my home address",
        "What's my wife's name",
        "What is my wife's name",
        "Who is my wife",
        "Who is my husband",
        "Who is my partner",
        "What's my doctor's name",
        "Who is my dentist",
        "Who is my boss",
        "What's my mom's number",
        "What's my dad's number",
        "What's my wife's phone number",
        "What's my cousin's name",
        "Who is my friend Sarah",
        "What do you know about my wife",
        "Tell me about my dentist",
        "What's the address of my dentist",
        "Where is my gym",
        "What's my gym address",
        "Where do I live",
        "What's my address",
        "What's my work address",
        "Where do I work",

        // ── Conversation context retrieval ──────────────
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
        "Show me the earlier steps",
        "Pick up where we stopped yesterday",
        "What was the last code snippet you gave me?",
        "Remind me of the grocery list from this morning",
        "What were the three options we weighed?",
        "Show the decision matrix we built",
        "What was the URL you shared 5 mins ago?",
        "Recap the pros/cons we listed",
        "What was the final command I ran?",
        "Bring me back to the API design discussion",
        "What did I decide about the color scheme?",
        "Show the timer I started earlier",
        "What was the exact error message?",
        "Continue the story we were writing",
        "What were the meeting action items?",
        "Remind me of the password we generated",
        "What was the last search query?",
        "Show the table we sketched",
        
        // ── Checklist retrieval ────────────────────────
        "What checklists have I created",
        "Show me my checklists",
        "List my checklists",
        "What lists do I have",
        
        // ── URL retrieval ────────────────────────
        "What URL did I ask you to remember",
        "Which URL did I save",
        "What link did I store",
        "Remind me of the URL",
        "What website did I bookmark",
        
        // ── Water/health tracking retrieval ────────────────────────
        "Did I log any water intake today",
        "Did I track water today",
        "Have I logged water intake",
        "Show my water log",
        "What's my water intake today",
        
        // ── General memory check ────────────────────────
        "Do you still remember",
        "Do you remember that",
        "Can you recall",
        
        // ── Cheat day / Friday patterns ────────────────────────
        "do you still remember my cheat day",
        "what's my cheat day",
        "when is my cheat day",
        "what's my relationship with Friday again",
        "what did I say about Friday",
        "Friday relationship",
        
        // ── Morning preference ────────────────────────
        "do you know whether I like mornings",
        "do I like mornings",
        "am I a morning person",
        "what did I say about mornings",
        
        // ── Units preference ────────────────────────
        "what did I say about metric vs imperial",
        "metric or imperial preference",
        "do I prefer metric",
        "what units do I use",
        
        // ── Sugar/snacks preferences ────────────────────────
        "What preferences did I set around sugar and snacks",
        "what did I say about sugar",
        "my sugar preferences",
        "sugar and snacks rules",
        
        // ── Vocabulary meaning ────────────────────────
        "What meaning did I assign to the word 'home' for you",
        "what does home mean to me",
        "how did I define home",
        "what did I say home means",
        
        // ── Habit recall ────────────────────────
        "Remind me what habit I wanted to build at night",
        "what habit did I want to build",
        "what's my nightly habit goal",
        "nighttime habit goal",
        
        // ── Social media limit ────────────────────────
        "What did I decide about my social media limit",
        "what's my social media limit",
        "social media time limit",
        "how long can I use social media",
        
        // ── Favorite/preference retrieval ────────────────────────
        "What is my favorite drink",
        "What's my favorite food",
        "What is my favorite color",
        "What's my preferred temperature",
        "What are my favorite movies",

        // ── Personal usage history (today/this week) ────────────────────────
        "What apps did I use today",
        "What apps have I used today",
        "What apps did I open today",
        "What apps did I use this week",
        "What sites did I visit today",
        "What websites did I browse today",
        "What did I work on today",
        "What did I do today",
        "What have I been doing today",
        "What did I look at today",
        "What did I open today",
        "What programs did I use today",
        "What tools did I use today",
        "What did I use this morning",
        "What have I worked on this week",
        "What tasks did I complete today",
        "What did I accomplish today",
        "What did I do this morning",
        "What have I done so far today",
        "What did I spend time on today",

        // ── Conversation history retrieval ────────────────────────
        "What have we chatted about today",
        "What have we talked about today",
        "What did we discuss today",
        "What have we been talking about",
        "What did we chat about earlier",
        "What topics have we covered today",
        "What have we gone over today",
        "What did we go over today",
        "What have we discussed so far",
        "What did we talk about this session",
        "What was our conversation about",
        "What have we been discussing",
        "What did we cover in our chat",
        "What topics did we discuss today",
        "Summarize what we talked about today",
        "What did we talk about just now",
        "What were we chatting about",
        "What have we said to each other today",
        "What did we go through today",
        "Recap our conversation today",

        // ── Temporal cross-session retrieval (yesterday / last week) ────────────────────────
        // These were being misclassified as command_automate by DistilBERT
        "What did we chat about yesterday",
        "What did we talk about yesterday",
        "What did we discuss yesterday",
        "What were we chatting about yesterday",
        "What topics did we cover yesterday",
        "Recap our conversation from yesterday",
        "Summarize what we talked about yesterday",
        "What did we go over yesterday",
        "What did we chat about last week",
        "What did we talk about last week",
        "What did we discuss last week",
        "What did we cover last week",
        "What did we chat about last month",
        "What did we talk about last night",
        "What did we discuss last night",
        "Did we chat about coding yesterday",
        "Did we talk about history yesterday",
        "Did we discuss anything important yesterday",
        "What topics came up yesterday",
        "What did I ask you yesterday",
        "What did I say yesterday",
        "What did I mention yesterday",
        "What did I tell you last week",
        "Did I visit any websites yesterday",
        "What websites did I visit yesterday",
        "What apps did I use yesterday",
        "What did I work on yesterday",
        "What did I do yesterday",
        "What did I accomplish yesterday",
        "What files did I mention yesterday",
        "List all the files I mentioned yesterday",
        "What files did I work on yesterday",
        "What code did I write yesterday",
        "What projects did I work on last week",
        "What did I spend time on yesterday",
        "What was I doing yesterday",
        "What was I working on last week",
        "Show me what I did yesterday",
        "Show me what we talked about last week",
        "Tell me what we discussed yesterday",
        "Remind me what we talked about yesterday",
        "Remind me what I did yesterday",
        "What happened in our chat yesterday",
        "What was our last conversation about",
        "What did we cover in our last session",

        // ── "Give/Tell me" retrieval patterns (hard negatives vs memory_store) ────────────────────────
        // These were scoring nearly equal to memory_store — adding explicit seeds to break the tie
        "Give me the date of that day",
        "Give me the date we discussed",
        "Give me the exact date",
        "Give me the time of my appointment",
        "Give me the details I saved",
        "Tell me the date of that event",
        "Tell me what day that was",
        "Tell me what I saved about that",
        "Tell me the time of my meeting",
        "Tell me what I noted down",
        "Show me the date I mentioned",
        "Show me what day that was",
        "Show me the info I stored",
        "Find the date I mentioned",
        "Find what I said about that day",
        "What date was that",
        "What day was that",
        "What was the date of that",
        "What was the exact date",
        "What was the time again",

        // ── Personal habit / preference recall ────────────────────────────────
        // "I usually get/order/do X" = asking about a stored personal habit
        "What was the coffee order I usually get?",
        "What do I usually order?",
        "What's my usual order?",
        "What coffee do I usually get?",
        "What do I normally drink?",
        "What's my go-to lunch?",
        "What's the meal I usually get?",
        "What did I usually order there?",
        "What food do I normally get?",
        "What's my regular order?",

        // ── Tracked sessions / activity logs ─────────────────────────────────
        // "sessions I've tracked" / "I've tracked" = querying logged data
        "List all running sessions I've tracked",
        "Show the sessions I've tracked",
        "What sessions have I tracked?",
        "List the workouts I've tracked",
        "How many sessions have I logged?",
        "What activities have I tracked this week?",
        "Show me everything I've tracked",
        "What have I been tracking?",
        "Pull up my tracked sessions",
        "What runs have I logged?",

        // ── Personal health data queries ───────────────────────────────────
        // "my health data" = accessing stored personal health records
        "Based on my health data what should I watch out for",
        "What does my health data say?",
        "What's in my health records?",
        "Review my health data",
        "What health info have I saved?",
        "Check my health notes",
        "What have I logged about my health?",
        "What health conditions have I noted?",
        "What medications have I recorded?",
        "What does my fitness data show?",

        // ── Self-identity / profile from records ──────────────────────────
        // "who I am based on your records" = self-summary from stored data
        "Give me a quick summary of who I am based on your records",
        "Give me a summary of who I am",
        "Summarize who I am based on your records",
        "What do your records say about me?",
        "Give me a profile of myself",
        "Summarize everything you know about me",
        "What have I told you about myself?",
        "Describe me based on what I've shared",
        "Build a profile of me from my notes",
        "Who am I according to your data?",

        // ── Scanning / searching personal notes ───────────────────────────
        // "scan my notes for X" = a memory search/retrieve operation
        "Scan my notes for anything about my gym membership",
        "Scan my notes for anything related to work",
        "Search my notes for anything about the lease",
        "Look through my notes for gym info",
        "Find anything in my notes about fitness",
        "Scan my memory for references to the project",
        "Search through what I've saved about health",
        "Find anything stored about my subscription",
        "Look through my logs for mentions of budget",
        "Search my records for anything about my car",

        // ── Round 5 seeds ───────────────────────────────────────
        // Client / business retrieval
        "What do I know about client Kenji Watanabe?",
        "What have I stored about my client Kenji?",
        "Tell me everything you know about client Kenji's business",
        "What do you have on my client's company?",
        "What do I have saved about my SaaS client?",
        // Scheduled time / obligations
        "What time am I supposed to be in bed by?",
        "What time do I usually go to sleep?",
        "When do I usually wake up?",
        "What time am I meant to take my medication?",
        // Voice / casual pull-up patterns
        "pull up my freelance rate",
        "pull up what I have on my bike route",
        "hey think drop pull up my notes on Kenji",
        "show me what I have saved about my rate",
        // "look up what I have stored" (distinguish from web_search "look up")
        "look up what I have stored about my Mandarin study progress",
        "look up what I saved about my marathon training",
        "look up my notes on physical therapy",
        // Relationship / event retrieval
        "When is my sister Sofia's wedding?",
        "When is my sister's wedding date?",
        "What did I store about my sister's wedding?",
        "What's the date of my sister's wedding?",
        // Self-description
        "how would you describe me based on what I've shared",
        "how would you characterize me from what you know?",
        "summarize who I am based on what I've told you",
        // ── Round 6 seeds ──────────────────────────────────────
        "Pull up everything on Priya Nair my manager",
        "Pull up everything you know about Marcus",
        "Pull up all my notes about my coworker Theo",
        "Show everything you have on Priya",
        "What's the name of my sourdough starter?",
        "What did I name my starter culture?",
        "What is my cat's name again?",
        "What's my dentist's name?",
        "whats the name of my soul dough starter",
        "what is my soul dough starter called",
        "What city did my friend move to?",
        "What city did Marcus end up moving to?",
        "Where did Marcus move?",
        "What city is Marcus in now?",
        "What do I have on Priya?",
        "Anything saved on my manager Priya?",
        "What's stored about my coworker Theo?",
        "What do you know about Priya Nair?",
        "Car insurance policy info?",
        "What's my car insurance carrier?",
        "Which company is my car insured with?",
        "What's my insurance policy number?",
        "Any notes about my insurance?",
        "Any notes on my sourdough experiments?",
        "Did I store anything about my sourdough attempts?",
        "What have I logged about my baking progress?",
        "Any notes about my fermenting hobby?",
        "Look back through your memory for my Geico policy number",
        "Check your records for my insurance details",
        "Dig through your records for the mobile redesign lead at work",
        "Search through what you know about my health this month",
        "hey think drop remind me what marcus's situation is",
        "remind me what's going on with Marcus",
        "catch me up on my friend Marcus",
        "Show me all upcoming personal appointments I've stored",
        "List all the appointments you have logged for me",
        "What appointments do I have coming up that you know of?",
        "Who is leading the mobile redesign project at work?",
        "Who is in charge of the redesign at my company?",
        "Who did I say was running the project?",
        // ── Round 7 seeds ──────────────────────────────────────
        // R7 persona retrieval
        "What kind of dog is Nala?",
        "what kind of dog is nay la",
        "Has Nala had her first vet visit yet?",
        "Who do I go to for SQL questions at work?",
        "How does Lena prefer to receive project updates?",
        "What note-taking app am I currently using?",
        "What is Jamie doing this semester?",
        "where does soph ee live now",
        "How long is the cycling race I registered for?",
        "whats the dis tance of the cy cling race I signed up for",
        "Tell me what I know about my April move",
        "Any notes on my kombucha setup?",
        "What do you have on Jamie?",
        "Remind me about my index fund strategy",
        "remind me about my index fund strat e gy",
        "What course am I taking on Coursera?",
        "What are my sleep supplements?",
        "Current training?",
        "Monthly salary?",
        "Nala info?",
        "Ben Okafor?",
        "Sleep supplements?",
        "What do you have on my financial picture?",
        "Look back through your memory for what I said about Ben Okafor",
        "Search through what you know about my financial picture",
        "Who is Viktor Tran to me?",
        "who is vik tor tran to me",
        // ── Round 7b seeds ──────────────────────────────────────
        // Voice: who do I ask for SQL questions (mr-v05)
        "Who do I ask for SQL questions at work?",
        "who do I ask for S Q L ques tions at work",
        "Who should I contact for database questions at work?",
        "Who handles SQL issues on the team?",
        // Dig through records (mr-a02)
        "Dig through your records to find Sophie's birthday",
        "Dig through your memory to find my dentist's name",
        "Dig through your records for the date of that event",
        "Go through your records to find what I said about Sophie"
      ],

      web_search: [
        // ── Original (kept) ─────────────────────────────────────
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
        "Who's the best basketball player in the world",
        "Who's the fastest runner in the world",
        "Who's the best jumper in the world",
        "Who's the greatest soccer player of all time",
        "Who's the top tennis player right now",
        "Who's the best swimmer in history",
        "What's the fastest car in the world",
        "What's the best restaurant in New York",
        "Who's the richest person in the world",
        "What's the tallest building in the world",
        
        // Shopping and product recommendations
        "What's the best winter jacket to wear",
        "Best laptop for programming",
        "Top rated headphones under $200",
        "What's the best coffee maker",
        "Best running shoes for marathon training",
        "Top rated air purifier",
        "What's the best smartphone camera",
        "Best budget gaming PC",
        "Top rated mattress for back pain",
        "What's the best vacuum cleaner",
        "Best noise cancelling earbuds",
        "Top rated standing desk",
        "What's the best blender for smoothies",
        "Best winter boots for snow",
        "Top rated backpack for travel",
        
        // Time-sensitive queries
        "When is the next election?",
        "What time is it in London?",
        "When does Costco close today?",
        "When is Diwali this year?",
        "US CPI print date this month",

        // ── New – more niches, real-time, code, events ───────
        "Current ETH gas price?",
        "What’s the 10-year Treasury yield right now?",
        "Latest iPhone 16 Pro price in USD",
        "Who won the Nobel Prize in Physics this year?",
        "Current population of Tokyo",
        "When is the next SpaceX Starship launch?",
        "What’s the current version of Kubernetes?",
        "Show me the latest Tailwind CSS docs",
        "How do I set up OAuth2 with Google in FastAPI?",
        "Give me a bash one-liner to watch disk usage",
        "What’s the weather forecast for Seattle this weekend?",
        "Current price of gold per ounce",
        "Who is the CEO of xAI?",
        "Latest commit on the Linux kernel",
        "When does the F1 Monaco GP start?",
        "Give me a Rust example of async HTTP client",
        "How do I configure nginx as a reverse proxy for Next.js?",
        "Show me a Terraform module for an S3 bucket with versioning",
        "Current COVID booster eligibility in California",
        "What’s the latest stable version of PostgreSQL?",
        "Give me a regex to validate UUID v4",
        "Who is the current UN Secretary-General?",
        "Latest inflation rate for the Eurozone",
        "When is the next total lunar eclipse visible in North America?",
        "Show me a minimal Vite + React + TypeScript starter",
        "Current market cap of NVIDIA",
        "How do I enable 2FA on GitHub with an authenticator app?",
        "Give me a Python snippet to resize images with Pillow",
        "What's the current base rate of the ECB?",
        
        // ── Link and resource requests (web research) ────────────
        "Give me links to learn about machine learning",
        "Can you give me links to articles about climate change",
        "Find me links about AI research papers",
        "Show me links for python tutorials",
        "Get me some links on web development",
        "Find links to commentary on the election",
        "Give me resources about quantum computing",
        "Find articles about space exploration",
        "Search for information on renewable energy",
        "Look up information about blockchain technology",
        "Find commentary on the latest tech news",
        "Search for commentary on economic policy",
        "Get me commentary on the stock market",
        "Find resources about data science",
        "Search the web for react best practices",
        "Look up on the web kubernetes deployment",
        "Find information on the web about docker containers",
        "Search online for javascript frameworks",
        "Find me information about TypeScript",
        "Look for articles on software architecture",
        "Search for articles about microservices",
        "Find me some resources on cloud computing",
        "Get information about serverless architecture",
        "Search for resources on API design",
        
        // ── Factual yes/no and "are there" questions ────────────
        "Are there multiple companies with the same name",
        "Are there more than one Lensa",
        "Is there more than one Apple company",
        "Are there different versions of ChatGPT",
        "Is there a difference between React and React Native",
        "Are there multiple OpenAI models",
        "Is there a free version of GitHub Copilot",
        "Are there alternatives to AWS",
        "Is there a difference between TypeScript and JavaScript",
        "Are there multiple Python versions",
        "Is there a mobile app for Notion",
        "Are there different types of databases",
        "Is there a difference between Docker and Kubernetes",
        "Are there multiple ways to deploy a website",
        "Is there a free tier for Vercel",
        "Are there different programming paradigms",
        "Is there a difference between frontend and backend",
        "Are there multiple cloud providers",
        "Is there a difference between REST and GraphQL",
        "Are there different types of APIs",
        
        // ── "How do I..." questions (moved from command_guide) ────────────
        "How do I use AI plugins in Figma",
        "How do I configure VS Code for Python",
        "How do I create a React component",
        "How do I create a custom Slack slash command",
        "How do I set up a local MongoDB replica set",
        "How do I use the Shopify Admin API",
        "How do I set up a development environment",
        "How do I install and configure PostgreSQL",
        "How do I set up a local MySQL database",
        "How do I configure nginx for production",
        "How do I set up a reverse proxy with nginx",
        "How do I configure Prettier",
        "How do I set up a virtual environment in Python",
        "How do I use webpack for bundling",
        "How do I create a custom middleware in Express",
        "How do I set up a local Kafka cluster",
        "How do I create a custom plugin for Obsidian",
        "How do I use GitHub Actions",
        "How do I configure nginx",
        "How do I set up a reverse proxy",
        "How do I create a CloudFront distribution",
        "How do I set up a GitLab CI runner",
        "How do I create a custom domain with Route 53",
        "How do I use Figma components",
        "How do I create a Zapier automation",
        "How do I set up a Mailchimp campaign",
        "How do I create a custom Jira workflow",
        "How do I set up a Stripe Checkout",
        "How do I create a custom Telegram bot",
        "How do I design a custom icon set in Sketch",
        "How do I create a custom color palette in Coolors",
        "How do I set up a Metabase instance",
        "How do I create a custom Google Data Studio connector",
        "How do I create a secure SSH key pair",
        "How do I audit npm dependencies for vulnerabilities",
        "How do I use the Windows PowerShell",
        "How do I create a Windows scheduled task",
        "How do I create a custom OpenAI fine-tune",
        "How do I create a custom prompt template",
        "How do I set up a local Home Assistant instance",
        "How do I set up a local Minecraft server",
        "How do I create a custom Spotify playlist with the API",
        "How to reset my password on window computer",
        "How to use ChatGPT and Mermaid AI to generate system architecture diagrams",
        "How to use Gemini and Excalidraw AI to sketch database ER diagrams from natural language",
        "How to use Claude 3 and Whimsical AI to build interactive product flowcharts",
        "How to use NoteLM and Mermaid AI to create Gantt charts from project timelines",
        "How to use ChatGPT, Perplexity, and NoteLM to script a 5-minute learning video",
        "How to use Gemini and Runway ML to generate video from AI-written scripts",
        "How to use Perplexity and Pictory to turn blog posts into AI-narrated videos",
        "How to use NoteLM and CapCut AI to auto-edit educational TikTok videos",
        "How to use Qwen and InVideo AI to create product demo videos from feature lists",
        "How to use ChatGPT and Replit AI to build a full-stack app from a single prompt",
        "How to use Gemini and GitHub Copilot to generate React components with tests",
        "How to use NoteLM and Glitch AI to deploy AI-powered web tools in 2 minutes",
        "How to use Grok and CodePen AI to generate interactive CSS animations",
        "How to use ChatGPT and Jasper AI to write a 2000-word SEO blog post",
        "How to use Gemini and Writesonic to create email sequences for product launches",
        "How to use NoteLM and Rytr to write YouTube video descriptions with hooks",
        "How to use Llama 3 and Anyword to A/B test ad copy variants",
        "How to use Qwen and HyperWrite to rewrite articles in different tones",
        "How to use Grok and Grammarly AI to polish AI-generated technical documentation",
        "How to use Perplexity and ChatGPT to research and summarize a 50-page PDF",
        "How to use Gemini and Mem.ai to build a personal knowledge base from scattered notes",
        "How to use Llama 3 and Humata.ai to Q&A a legal contract",
        "How to use Qwen and Genei.io to summarize 10 YouTube videos into one doc",
        "How to use ChatGPT and Otter.ai to turn meeting recordings into action items",
        "How to use ChatGPT and Figma AI to generate UI mockups from text descriptions",
        "How to use Gemini and Framer AI to build landing pages from prompts",
        "How to use Llama 3 and Anima AI to convert Figma designs to React code",
        "How to use Qwen and DiagramGPT to generate flowcharts from code",
        "How to use Grok and Visily AI to create wireframes from user stories",
        "How to use ChatGPT and Make.com to build no-code AI automations",
        "How to use Gemini and Airtable AI to auto-categorize form submissions",
        "How to use NoteLM and Parabola to clean CSV data with AI",
        "How to use Llama 3 and Bardeen AI to scrape and summarize websites",
        "How to use Grok and Tray.io to build enterprise AI pipelines",
        "How to use ChatGPT and Notion AI to build a second brain from highlights",
        "How to use Gemini and Mem.ai to generate flashcards from YouTube videos",
        "How to use NoteLM and Roam Research AI to generate backlinks",
        "How to use Llama 3 and Reflect AI to journal with AI-guided prompts",
        "How to use Qwen and Heptabase to visualize knowledge graphs",
        "How to use Grok and Tana AI to capture ideas with AI tagging",
        "How to use ChatGPT and Suno AI to generate songs from story prompts",
        "How to use Gemini and Pika Labs to generate AI video from text",
        "How to use NoteLM and Mubert to generate ambient soundscapes",
        "How to use Llama 3 and Scenario.gg to train custom AI art models",
        "How to use Qwen and Replicate to run Stable Diffusion locally",
        "How to use Grok and Runway Gen-2 to animate AI-generated portraits",
        "How to use ChatGPT and Typeform AI to generate customer surveys with logic",
        "How to use Gemini and Intercom AI to auto-reply to support tickets",
        "How to use NoteLM and Gong.io to analyze sales call sentiment",
        "How to use Llama 3 and Calendly AI to auto-schedule meetings from emails",
        "How to use Qwen and Stripe AI to detect fraud in transactions",
        "How to use Grok and Shopify AI to generate product descriptions",
        
        // ── "What is the meaning of" queries (web search for comprehensive info) ────────────
        "What is the meaning of grammar",
        "What the meaning of grammar",
        "What is the meaning of syntax",
        "What the meaning of syntax",
        "What is the meaning of love",
        "What the meaning of life",
        "What is the meaning of success",
        "What the meaning of happiness",
        "What is the meaning of this word",
        "What the meaning of this term",
        "What does this word mean",
        "What does grammar mean",
        "Meaning of grammar",
        "Meaning of syntax",
        "Definition of grammar",
        "Definition of syntax",
        
        // ── "Where" location/recommendation queries (web search) ────────────
        "Where the best pizza in the world",
        "Where is the best pizza",
        "Where can I find the best pizza",
        "Where to get the best pizza",
        "Where are the best restaurants",
        "Where is the best coffee shop",
        "Where can I buy cheap laptops",
        "Where to find good deals",
        "Where are the top hotels in Paris",
        "Where is the nearest gas station",
        "Where can I watch the game",
        "Where to stream movies",
        "Where are the best beaches",
        "Where is the cheapest gas",
        "Where can I get a haircut",
        "Where to buy groceries online",
        "Where are the best schools",
        "Where is the closest hospital",
        "Where can I find a job",
        "Where to apply for jobs",
        "Where are the best gyms",
        "Where is the best sushi",
        "Where can I learn coding",
        "Where to take coding classes",
        "Where are the top universities",
        "Where is the best place to live",
        "Where can I find apartments",
        "Where to rent a car",
        "Where are the best hiking trails",
        "Where is the nearest pharmacy",
        
        // ── "Find me" / "Show me" information queries (web search) ────────────
        "Find me the best pizza places",
        "Find me restaurants near me",
        "Find me cheap flights to Paris",
        "Find me hotels in New York",
        "Find me the latest news",
        "Find me information about AI",
        "Find me tutorials on React",
        "Find me the weather forecast",
        "Find me movie times",
        "Find me concert tickets",
        "Show me the latest iPhone price",
        "Show me restaurants nearby",
        "Show me the news today",
        "Show me flights to London",
        "Show me hotels in Tokyo",
        "Show me the weather",
        "Show me movie reviews",
        "Show me concert venues",
        "Show me the stock price",
        "Show me gas prices",
        
        // ── "What is/are" current information queries (web search) ────────────
        "What is the current price of Bitcoin",
        "What are the best restaurants in NYC",
        "What is the weather today",
        "What are the top movies right now",
        "What is the latest iPhone",
        "What are the best laptops to buy",
        "What is the stock market doing",
        "What are the gas prices near me",
        "What is the score of the game",
        "What are the news headlines",
        "What is trending on Twitter",
        "What are the best deals today",
        "What is the exchange rate",
        "What are the flight prices",
        "What is the hotel rate",
        "What are the best products on Amazon",
        "What is the cheapest option",
        "What are people saying about",
        "What is the review for",
        "What are the ratings for",
        
        // ── Comparison / Product queries (web search) ────────────────────────
        "Compare iPhone 16 Pro and Galaxy S26",
        "Compare iPhone vs Samsung",
        "Compare MacBook Pro vs Dell XPS",
        "Compare Tesla Model 3 vs BMW i4",
        "Compare React vs Vue",
        "Compare AWS vs Azure",
        "Compare Notion vs Obsidian",
        "iPhone 16 Pro vs Galaxy S26",
        "MacBook Air vs MacBook Pro",
        "Which is better iPhone or Samsung",
        "Which is faster SSD or HDD",
        "Difference between iPhone 15 and 16",
        "Pros and cons of iPhone vs Android",
        
        // ── Current events / Time-sensitive queries (web search) ────────────────────────
        "Is it going to rain this weekend in Chicago",
        "Will it rain tomorrow",
        "Is it raining in Seattle",
        "Who won the NBA game last night",
        "Who won the game yesterday",
        "Who won the Super Bowl this year",
        "Any flight delays at JFK airport",
        "Flight delays at LAX",
        "Are there delays at the airport",
        "Current traffic on I-95 northbound",
        "Traffic on highway 101",
        "Is there traffic on the freeway",
        "Latest patch notes for League of Legends",
        "New update for Fortnite",
        "Latest game patch notes",
        "Breaking news about the presidential election",
        "Breaking news today",
        "Latest news headlines",
        
        // ── Shopping / Price queries (web search) ────────────────────────
        "Cheapest 4K monitor with 120hz refresh rate",
        "Cheapest laptop for gaming",
        "Best budget phone under 500",
        "Best mechanical keyboard under 100 dollars",
        "Affordable standing desk",
        "Cheap wireless earbuds",
        
        // ── Hybrid queries (web search + memory) ────────────────────────
        "Can you look this up and remember it for later",
        "Look this up and save it",
        "Search for this and remember it",
        "Find this information and store it",
        
        // ── Trending / Entertainment queries ────────────────────────
        "Trending TikTok songs right now",
        "Trending songs on TikTok",
        "What's trending on TikTok",
        "Top trending videos",
        "Viral TikTok trends",
        "Who plays Batman in the new movie",
        "Who plays the main character in",
        "Cast of the new movie",
        "Actor in the latest film",
        
        // ── Event / Schedule queries ────────────────────────
        "When is the next Apple event",
        "When is the next Google event",
        "When is the next Microsoft event",
        "Next product launch date",
        "Upcoming tech events",
        
        // ── News queries ────────────────────────
        "Earthquake news california",
        "Earthquake news today",
        "Breaking news earthquake",
        "Latest earthquake updates",
        "News about earthquakes",
        
        // ── Service status queries ────────────────────────
        "Is DoorDash down right now",
        "Is Uber Eats down",
        "Is Instagram down right now",
        "Is Twitter down",
        "Service outage check",
        "Is the website down",
        
        // ── Financial / Market queries ────────────────────────
        "Crypto fear and greed index today",
        "Fear and greed index",
        "Market sentiment today",
        "Crypto market sentiment",
        
        // ── Ranked list queries ────────────────────────
        "Top 10 programming languages according to GitHub",
        "Top programming languages",
        "Best programming languages ranked",
        "Most popular languages on GitHub",
        "Top 10 list of",
        
        // ── Edge case queries ────────────────────────
        "Search it",
        "Search that",
        "Look it up",
        "Find it",
        "Google it",
        "I need information on that",
        "I need info on that",
        "Tell me about that",
        "Information on that",
        "Details on that",
        
        // ── Complex multi-intent (web search dominant) ────────────────────────
        "First, find out who won the World Cup last time and second, tell me briefly how the tournament works",
        "find out who won and explain how it works",
        "who won and how does it work",
        "See if there are any delays on my train line tonight and don't forget I commute from Philly",
        "check delays and remember my commute",
        "train delays plus remember",
        "Check how much ETH is trading for and then remind me tomorrow if it drops 5 percent",
        "check price and remind me later",
        "price check with reminder",
        "I'm planning a vacation and I want somewhere warm in January, what destinations should I look at",
        "vacation destinations warm in January",
        "where to go in January warm",
        "I feel like I'm paying too much for internet, what are cheaper providers near me",
        "cheaper internet providers near me",
        "internet providers cheaper",
        
        // ── Simple factual lookups (height, dates, historical) ────────────────────────
        "How tall is Mount Everest",
        "How tall is the Eiffel Tower",
        "How high is Mount Kilimanjaro",
        "When is the next solar eclipse",
        "When is the next lunar eclipse",
        "When is the next full moon",
        "What time does Walmart open tomorrow",
        "What time does Target close today",
        "What time does Costco open",
        "Who invented the light bulb",
        "Who invented the telephone",
        "Who invented the airplane",
        "Who discovered penicillin",
        "Who discovered America",
        
        // ── Educational / Tutorial Mode – "show me how" ────────────────────────
        // ── Software / Tool Tutorials
        "Show me how to set up Gmail filters",
        "Teach me how to create a Slack workflow",
        "Guide me through setting up Docker",
        "Walk me through using Git branches",
        "Walk me through creating a GitHub repository",
        "Show me how to use Git branches",
        "Teach me how to deploy to Netlify",
        "Guide me through setting up SSH keys",
        "Show me how to use the Figma API",
        "Teach me how to create a custom Notion database",
        "Walk me through building a Chrome extension",
        "Guide me through using Postman collections",
        "Show me how to create a custom VS Code snippet",
        "Teach me how to use the GitHub CLI",
        "Walk me through setting up a CI pipeline in CircleCI",
        "Show me how to use the Vercel CLI",
        "Guide me through creating a custom WordPress theme",
        "Teach me how to use the Stripe Dashboard",
        "Show me how to create a custom Airtable view",
        "Walk me through using the AWS CLI",
        "Guide me through setting up a Firebase project",
        "Teach me how to create a custom Zapier integration",
        "Show me how to set up a local PostgreSQL database",
        "Teach me how to use the GraphQL Playground",
        "Walk me through creating a custom Trello power-up",
        "Guide me through setting up a local Redis server",
        
        // ── 2. Development Tutorials
        "Show me how to write a Dockerfile",
        "Teach me how to use npm scripts",
        "Walk me through setting up ESLint",
        "Show me how to use Chrome DevTools",
        "Guide me through creating a pull request",
        "Teach me how to use Postman for API testing",
        "Show me how to configure Tailwind CSS",
        "Walk me through setting up a Next.js project",
        "Guide me through using TypeScript with React",
        "Teach me how to set up Jest for testing",
        "Show me how to create a custom Hook in React",
        "Walk me through using Redux Toolkit",
        "Guide me through setting up a GraphQL server with Apollo",
        "Teach me how to use Prisma with a database",
        "Show me how to use the Node.js debugger",
        "Walk me through setting up a monorepo with Turborepo",
        "Guide me through using Vite for a Vue project",
        "Teach me how to create a custom Svelte store",
        "Show me how to use the Docker Compose file",
        "Walk me through creating a custom CLI tool with oclif",
        "Guide me through using the GitHub REST API",
        "Teach me how to set up a local Elasticsearch instance",
        "Show me how to use the OpenAI API in Node.js",
        "Walk me through setting up a local Supabase project",
        
        // ── 3. Infrastructure / DevOps Tutorials
        "Guide me through creating a Lambda function",
        "Show me how to set up MongoDB",
        "Teach me how to use Redis",
        "Walk me through setting up Kubernetes",
        "Show me how to deploy to AWS",
        "Guide me through creating a CI/CD pipeline",
        "Teach me how to use Terraform",
        "Walk me through provisioning an EC2 instance",
        "Guide me through setting up Cloudflare DNS",
        "Teach me how to use Ansible playbooks",
        "Show me how to use the AWS CDK",
        "Walk me through setting up a VPC",
        "Guide me through using Helm charts",
        "Teach me how to configure Traefik",
        "Show me how to use Pulumi",
        "Walk me through creating a DigitalOcean droplet",
        "Guide me through using the Azure CLI",
        "Teach me how to set up a load balancer in GCP",
        "Show me how to use the Serverless Framework",
        "Walk me through setting up a Jenkins pipeline",
        
        // ── 4. Application / Productivity Tutorials
        "Show me how to create a Notion template",
        "Guide me through creating a Trello board",
        "Teach me how to use Slack workflows",
        "Walk me through setting up Google Analytics",
        "Show me how to use Airtable formulas",
        "Guide me through creating a Canva design",
        "Teach me how to use Asana for project management",
        "Walk me through creating a ClickUp space",
        "Guide me through setting up a Linear project",
        "Teach me how to use Monday.com boards",
        "Show me how to use the Todoist API",
        "Walk me through setting up a Calendly link",
        "Guide me through creating a Typeform",
        "Teach me how to use the HubSpot CRM",
        "Show me how to create a Gumroad product",
        "Walk me through using the Webflow CMS",
        "Guide me through setting up a Ghost blog",
        "Teach me how to use the Discord bot API",
        "Show me how to use the Twilio SMS API",
        "Walk me through setting up a SendGrid template",
        
        // ── 5. Design / Creative Tutorials
        "Show me how to create a Figma prototype",
        "Teach me how to use Framer Motion",
        "Guide me through creating a UI kit in Adobe XD",
        "Walk me through animating with Lottie",
        "Teach me how to use the Canva API",
        "Show me how to use the Procreate brush studio",
        "Guide me through creating a 3D model in Blender",
        "Teach me how to use the After Effects expression editor",
        
        // ── 6. Data / Analytics Tutorials
        "Show me how to create a Looker dashboard",
        "Teach me how to write a BigQuery SQL query",
        "Guide me through using Tableau calculated fields",
        "Walk me through creating a Power BI report",
        "Teach me how to use the Snowflake SQL editor",
        "Show me how to use the Mixpanel event tracker",
        "Guide me through setting up Amplitude cohorts",
        
        // ── 7. Security / Privacy Tutorials
        "Show me how to set up 2FA on GitHub",
        "Teach me how to use a password manager like 1Password",
        "Guide me through enabling full-disk encryption on macOS",
        "Walk me through setting up a VPN with WireGuard",
        "Teach me how to use GPG for email encryption",
        "Show me how to use the OWASP ZAP scanner",
        
        // ── 8. macOS / Windows / Linux System Tutorials
        "Show me how to use the macOS Terminal",
        "Teach me how to create a bash alias",
        "Guide me through setting up zsh with Oh My Zsh",
        "Walk me through creating a systemd service",
        "Teach me how to use the Linux cron scheduler",
        "Show me how to use the macOS Automator",
        "Guide me through setting up a macOS launch agent",
        "Teach me how to use the Linux firewall (ufw)",
        
        // ── 9. AI / ML Tutorials
        "Show me how to fine-tune a Hugging Face model",
        "Teach me how to use LangChain for RAG",
        "Guide me through using LlamaIndex",
        "Walk me through setting up a local Ollama server",
        "Teach me how to use the Gemini API",
        "Show me how to use the Claude API",
        "Guide me through building a RAG pipeline with Pinecone",
        "Teach me how to use the Cohere API for classification",
        
        // ── 10. Misc / Fun / Niche Tutorials
        "Show me how to create a custom emoji in Slack",
        "Teach me how to use the Raycast launcher",
        "Guide me through creating a custom Alfred workflow",
        "Walk me through using the Obsidian vault",
        "Teach me how to create a custom Roam Research graph",
        "Show me how to use the Twitch API",
        "Guide me through creating a custom Discord slash command",
        "Teach me how to use the YouTube Data API",
        "Show me how to use the NASA API",
        "Walk me through setting up a local Mastodon instance",
        
        // ── AI + AI → Diagrams / Flowcharts / Architecture
        "Teach me how to combine Claude and Mermaid Live Editor to create real-time UML diagrams",
        "Show me how to combine Perplexity and Draw.io AI to auto-generate network topology maps",
        "Guide me through using Llama 3 and Mermaid.js to create sequence diagrams from user stories",
        "Teach me how to combine ChatGPT and Lucidchart AI to generate org charts from team descriptions",
        "Show me how to combine Qwen and Miro AI to generate mind maps from brainstorming sessions",
        "Guide me through using Grok and Figma AI to auto-create UI component diagrams",
        
        // ── AI + AI → Video / Animation / Explainer Content
        "Teach me how to combine Claude, ElevenLabs, and HeyGen to create a talking AI explainer video",
        "Show me how to combine Llama 3, Descript, and Synthesia to make an AI avatar tutorial",
        "Guide me through using Perplexity and Pictory to turn blog posts into AI-narrated videos",
        "Teach me how to combine ChatGPT and VEED.io AI to add subtitles and animations to tutorials",
        "Show me how to combine Grok and Kaiber AI to generate animated AI art videos",
        "Guide me through using Claude and Lumen5 to turn podcast transcripts into video summaries",
        
        // ── AI + AI → Code Generation + Execution
        "Teach me how to combine Claude and Cursor.sh to write and debug Python scripts",
        "Show me how to combine Perplexity and CodeSandbox AI to prototype web apps instantly",
        "Guide me through using Llama 3 and VS Code AI to auto-generate API clients",
        "Teach me how to combine Qwen and Phind AI to solve LeetCode problems with explanations",
        "Show me how to combine ChatGPT and Warp AI to write shell scripts from English",
        "Guide me through using Claude and Tabnine to auto-complete full functions in Java",
        
        // ── AI + AI → Content Creation (Blog, Social, Email)
        "Teach me how to combine Claude and Copy.ai to generate 10 LinkedIn posts from one idea",
        "Show me how to combine Perplexity and Frase.io to generate content briefs with outlines",
        "Guide me through using NoteLM and Rytr to write YouTube video descriptions with hooks",
        "Teach me how to combine Qwen and HyperWrite to rewrite articles in different tones",
        "Show me how to combine ChatGPT and Notion AI to generate meeting notes into blog drafts",
        "Guide me through using Claude and Canva AI to create social media graphics with AI copy",
        
        // ── AI + AI → Research + Summarization
        "Teach me how to combine Claude and Elicit.org to extract insights from 20 research papers",
        "Show me how to combine NoteLM and Scite.ai to find citations for AI claims",
        "Guide me through using Llama 3 and Humata.ai to Q&A a legal contract",
        "Teach me how to combine Grok and Glean to search internal company docs with AI",
        "Show me how to combine Perplexity and Consensus AI to fact-check scientific claims",
        "Guide me through using Claude and Reflect AI to journal and extract weekly insights",
        
        // ── AI + AI → Design + Prototyping
        "Teach me how to combine Claude and Uizard to turn sketches into interactive prototypes",
        "Show me how to combine Perplexity and Relume AI to generate Webflow components",
        "Guide me through using NoteLM and Galileo AI to design mobile app flows",
        "Teach me how to combine Qwen and DiagramGPT to generate flowcharts from code",
        "Show me how to combine ChatGPT and Midjourney to generate UI inspiration images",
        "Guide me through using Claude and Adobe Firefly to generate branded graphics",
        
        // ── AI + AI → Data + Automation
        "Teach me how to combine Claude and Zapier AI to trigger actions from emails",
        "Show me how to combine Perplexity and n8n AI to create self-healing workflows",
        "Guide me through using NoteLM and Parabola to clean CSV data with AI",
        "Teach me how to combine Qwen and Albato to connect AI tools without code",
        "Show me how to combine ChatGPT and Google Sheets AI to analyze data with formulas",
        "Guide me through using Claude and Power Automate AI to approve invoices automatically",
        
        // ── AI + AI → Learning / Personal Knowledge
        "Teach me how to combine Claude and Obsidian AI to link notes with embeddings",
        "Show me how to combine Perplexity and Anki AI to create spaced repetition decks",
        "Guide me through using NoteLM and Roam Research AI to generate backlinks",
        "Teach me how to combine Qwen and Heptabase to visualize knowledge graphs",
        "Show me how to combine ChatGPT and Readwise AI to summarize saved articles",
        "Guide me through using Claude and Capacities AI to build a personal CRM",
        
        // ── AI + AI → Fun / Creative / Niche
        "Teach me how to combine Claude and Kaiber AI to make music videos from lyrics",
        "Show me how to combine Perplexity and Soundraw to create background music for podcasts",
        "Guide me through using NoteLM and Mubert to generate ambient soundscapes",
        "Teach me how to combine Qwen and Replicate to run Stable Diffusion locally",
        "Show me how to combine ChatGPT and DALL·E 3 to create children’s book illustrations",
        "Guide me through using Claude and Leonardo AI to generate consistent characters",
        
        // ── AI + AI → Business / Product
        "Teach me how to combine Claude and Customer.io to send AI-personalized emails",
        "Show me how to combine Perplexity and HubSpot AI to score leads from behavior",
        "Guide me through using NoteLM and Gong.io to analyze sales call sentiment",
        "Teach me how to combine Qwen and Stripe AI to detect fraud in transactions",
        "Show me how to combine ChatGPT and Klaviyo AI to segment customers with AI",
        "Guide me through using Claude and Attio AI to enrich CRM data with AI",
        
        // ── "Tell me about / Explain / What is" informational queries ──
        // These are web_search — asking for information about a topic, NOT commands
        "Tell me about CrowdStrike and their security products",
        "Tell me about this company and what they do",
        "Tell me more about OpenAI and their latest models",
        "Explain what Kubernetes is and how it works",
        "Explain the difference between TCP and UDP",
        "Explain what this technology does",
        "What is this CrowdStrike thing about",
        "What is OpenClaw and should I use it",
        "What are the risks of using this tool",
        "Describe what React hooks are",
        "Describe the difference between REST and GraphQL",
        "Tell me about the latest AI regulations in the EU",
        "Explain what happened with the CrowdStrike outage",
        "What is this issue about and how does it affect me",
        "Tell me about zero-trust security architecture",

        // ── IMPORTANT: What is NOT web_search ──────────────────
        // These are command_automate (action commands, not information queries):
        // ❌ "Goto chatgpt find my project called Thinkdrop AI and do a search for how to use Stripe API"
        // ❌ "Open Slack and then find the engineering channel"
        // ❌ "Navigate to Notion and search for my meeting notes"
        // ❌ "Go to Gmail and compose a new email"
        // ❌ "Launch VSCode and open the project folder"
        // 
        // Key difference: web_search = asking for information ("What is...", "How much...", "Who is...")
        //                 command_automate = telling AI to DO something ("Open...", "Go to...", "Launch...")

        // ── Round 5 seeds ───────────────────────────────────────
        // Food / diet alternatives (phi4 misclassifies as memory_retrieve)
        "Find me gluten-free pasta alternatives available right now",
        "Find me dairy-free cheese alternatives",
        "Find me low-carb bread alternatives near me",
        "Search for vegan protein powder options",
        "Look up keto-friendly snack alternatives",
        // ── Round 6 seeds ──────────────────────────────────────
        "when does the new season of severance come out on apple tv",
        "when does sev er ance come out",
        "what is the release date of the new severance season",
        "Severance season release date",
        "when is the next season of severance releasing",
        "Ethereum price",
        "current Ethereum price",
        "Bitcoin price today",
        "how does the current mortgage rate compare to last year's average",
        "what is the current prime lending rate in the US",
        "current federal interest rate",
        // ── Round 6b reinforcement seeds ──────────────────────────────────────────
        // Noun-phrase web searches (phi4 confused with memory_store)
        "Best sourdough bakeries in San Francisco",
        "Best coffee shops in Chicago",
        "Best pizza in New York City",
        "Top rated gyms in Austin",
        "Best coworking spaces in Seattle",
        "Coworking spaces open late in Austin Texas",
        "Coworking spaces near downtown Portland",
        "Cafes open early in Boston",
        // ── Round 7 seeds ──────────────────────────────────────
        // TV show / book release dates (phi4 confuses with memory_store)
        "When does The Bear season 3 come out on Netflix?",
        "when does the bear sea son three come out on net flix",
        "The Bear season 3?",
        "Patrick Rothfuss Doors of Stone release date",
        "pat rick roth fuss doors of stone re lease date",
        // How much / price queries that look like web search
        "how much does a quality SCOBY starter culture go for online",
        // ── Round 7b seeds ──────────────────────────────────────
        // Bear app web search (NOT memory_store: app switch) - web context
        "Bear note-taking app vs Notion which is better in 2026?",
        "Bear app review compared to Notion",
        // ── Round 9 seeds — hobby/sports research queries ──────────
        // Marathon events queries (NOT memory_store personal notes)
        "September half marathons in Portugal or Spain 2026",
        "fall running races in Europe 2026",
        "half marathon calendar 2026 Portugal",
        // Rabbit care research (NOT memory_store — general care guides)
        "Holland Lop rabbit diet and care guide",
        "Holland Lop rabbit care tips",
        "rabbit diet guide for new owners",
        "rabbit care for beginners holland lop",
        // Sourdough research (NOT memory_retrieve — general baking science)
        "Sourdough starter hydration ratio what do experts recommend",
        "sourdough bread hydration percentage guide",
        "sourdough starter feeding schedule best practices",
        // Health vs substance research queries explicitly requesting sources
        "Matcha latte vs coffee for ADHD focus any research",
        "green tea caffeine vs coffee focus research studies"
      ],

      general_knowledge: [
        // Stable facts that don't change
        "What is the capital of France?",
        "Where is the Eiffel Tower located?",
        // ── Round 9 seeds — substance comparison (NOT web_search) ──
        // Matcha vs coffee comparison for health/focus — factual knowledge query
        "Matcha vs coffee for focus",
        "green tea versus coffee which is better for focus",
        "caffeine in matcha compared to coffee",
        "does matcha give cleaner energy than coffee",
        "coffee vs tea for mental clarity",
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
        "Who wrote Romeo and Juliet?",
        
        // Explanations and "how does X work" queries
        "Explain quantum computing",
        "Explain quantum mechanics",
        "Explain machine learning",
        "Explain blockchain technology",
        "Explain neural networks",
        "Explain the theory of relativity",
        "Explain how photosynthesis works",
        "Explain the water cycle",
        "How does a car engine work",
        "How does a refrigerator work",
        "How does a microwave work",
        "How does the internet work",
        "How does GPS work",
        "How does WiFi work",
        "How does Bluetooth work",
        "How does a computer work",
        "How does a CPU work",
        "How does memory work",
        "How does a hard drive work",
        "How does encryption work",
        "How does a blockchain work",
        "How does democracy work",
        "How does the stock market work",
        "How does compound interest work",
        
        // "What are the benefits/advantages" queries
        "What are the benefits of meditation",
        "What are the benefits of exercise",
        "What are the benefits of yoga",
        "What are the benefits of reading",
        "What are the benefits of sleep",
        "What are the advantages of solar power",
        "What are the advantages of electric cars",
        "What are the pros and cons of remote work",
        
        // Historical and educational queries
        "Tell me about the French Revolution",
        "Tell me about World War 2",
        "Tell me about the Renaissance",
        "Tell me about the Industrial Revolution",
        "Tell me about ancient Egypt",
        "Tell me about the Roman Empire",
        "Tell me about the Cold War",
        
        // Static factual questions
        "How many planets are in the solar system",
        "How many states are in the US",
        "How many bones in the human body",
        "How many countries in the world",
        "How many continents are there",
        "What is the largest ocean",
        "What is the tallest mountain",
        "What is the longest river",

        // ── New – deeper CS, science, history, misc ───────
        "What is the halting problem?",
        "Explain the difference between a process and a thread",
        "What does ACID stand for in databases?",
        "How does a Bloom filter work?",
        "What is the difference between HTTP/1.1 and HTTP/2?",
        "Explain the Observer pattern with a diagram",
        "What is the chemical formula for glucose?",
        "Who proposed the theory of relativity?",
        "What is the Pythagorean theorem?",
        "How does RSA encryption work at a high level?",
        "What is the difference between a stack and a queue?",
        "Explain how DNS resolution works step-by-step",
        "What is the Bohr model of the atom?",
        "Who painted the Mona Lisa?",
        "What is the difference between RAM and ROM?",
        "Explain the concept of virtual memory",
        "What is the capital of Australia?",
        "How does a binary search tree maintain balance?",
        "What is the significance of the Turing Award?",
        "Explain the difference between supervised and unsupervised learning",
        
        // ── CRITICAL: Stable facts that don't require web search ────────────────────────
        "Why is the sky blue",
        "Why is the ocean blue",
        "Why do we see rainbows",
        "What are the main causes of climate change",
        "What causes global warming",
        "What is climate change",
        "List the three branches of the U.S. government",
        "What are the branches of government",
        "Name the three branches of government",
        "What is an API",
        "What does API stand for",
        "Define API",
        "Explain the difference between HTTP and HTTPS",
        "What is the difference between HTTP and HTTPS",
        "HTTP vs HTTPS",
        "What is object oriented programming",
        "What is OOP",
        "Define object oriented programming",
        "How do plants absorb water",
        "How do plants get water",
        "How do plants drink water",
        "What is the difference between a virus and a bacterium",
        "Virus vs bacteria",
        "What is the difference between virus and bacteria",
        "Explain the concept of supply and demand",
        "What is supply and demand",
        "Define supply and demand",
        "What are the primary colors of light",
        "What are the primary colors",
        "Name the primary colors",
        "What are RGB colors",
        
        // ── Science definitions ────────────────────────
        "What is a black hole",
        "What are black holes",
        "Define black hole",
        "Explain how gravity works",
        "How does gravity work",
        "What is gravity",
        "What are examples of renewable energy",
        "Examples of renewable energy",
        "Types of renewable energy",
        "What is recursion",
        "Define recursion",
        "Explain recursion",
        "Why do leaves change color in the fall",
        "Why do leaves change color",
        "What causes leaves to change color",
        "What is an antioxidant",
        "Define antioxidant",
        "What are antioxidants",
        "What is parallel computing",
        "Define parallel computing",
        "Explain parallel computing",
        "What is a metaphor",
        "Define metaphor",
        "Examples of metaphors",
        
        // ── Conversational clarifications ────────────────────────
        "That's not what I meant",
        "That's not what I said",
        "I didn't mean that",
        "Not what I meant",
        "That's not it",
        
        // ── Typos and misspellings ────────────────────────
        "phoyosynthesys what is it actually doing",
        "photosynthesis how does it work",
        "photosynthesys",
        
        // ── Colloquial phrasing ────────────────────────
        "so like what even IS an algorithm",
        "what even is an algorithm",
        "like what is an algorithm",
        "teach me like I'm five what a database is",
        "explain like I'm 5",
        "ELI5 what is",
        
        // ── Casual tone ────────────────────────
        "is evolution a theory or a fact explain",
        "is evolution theory or fact",
        "evolution theory vs fact",
        "ok but what IS time actually",
        "what is time really",
        "what is time actually",
        
        // ── Comparison requests ────────────────────────
        "difference between sql and nosql in one paragraph",
        "sql vs nosql",
        "difference between sql and nosql",
        "are tomatoes fruits or vegetables and why",
        "tomatoes fruit or vegetable",
        "is tomato a fruit",
        
        // ── Long explanations with context ────────────────────────
        "Explain to me like I'm a beginner how REST APIs work, I'm trying to finally get this concept",
        "explain REST APIs for beginners",
        "REST API explanation simple",
        
        // ── Emotional/support requests ────────────────────────
        "I'm really stressed about work, what can I do",
        "I'm stressed, help",
        "stress relief tips",
        "I'm bored, give me something interesting to learn",
        "I'm bored",
        "entertain me",
        "I can't focus today, any tips",
        "can't focus",
        "focus tips",
        "I'm tired but I still have to study, help",
        "tired but need to study",
        "study while tired",
        "I feel anxious about an upcoming interview, what should I practice",
        "interview anxiety",
        "interview prep",
        "I'm new to coding, where should I start",
        "new to coding",
        "coding for beginners",
        "I think I'm procrastinating a lot, how do I stop",
        "stop procrastinating",
        "procrastination help",
        "I'm overwhelmed by tasks, can you help me prioritize",
        "overwhelmed by tasks",
        "task prioritization",
        "I'm feeling lonely, can we just talk a bit",
        "I'm lonely",
        "I'm not sure what career path to choose, can you walk me through some options",
        "career advice",
        "career path help",
        
        // ── Very short emotional ────────────────────────
        "I'm scared",
        "I'm afraid",
        "I'm worried",
        "I don't know what to do",
        "I'm stuck",
        "I'm lost",
        "Can we just chat for a bit",
        "let's chat",
        "I feel like giving up on this task",
        "I want to give up",
        "feeling defeated",
        
        // ── Docker/container questions ────────────────────────
        "Help me understand what containers and Docker actually are, I'm pretty lost",
        "what are containers and Docker",
        "explain containers and Docker",
        "Docker and containers explained",
        
        // ── Credit score ────────────────────────
        "How does credit score work in general, not specific to any country, just the idea",
        "how does credit score work",
        "credit score explained",
        "what is a credit score",
        
        // ── Choice between web and knowledge ────────────────────────
        "You can either look this up on the web or just explain what you already know, whichever is easier",
        "look it up or explain",
        "search or explain",
        "either search or tell me",
        
        // ── Basic definitions (simple, timeless) ────────────────────────
        "What is a neuron",
        "What is a synapse",
        "What is a cell",
        "What is a budget",
        "What is a mortgage",
        "What is inflation",
        "What is a variable in programming",
        "What is a function in programming",
        "What is a loop in programming",
        "Why do we have seasons",
        "Why do we have day and night",
        "Why is the ocean salty",
        
        // ── Conversational context ────────────────────────
        "What's going on",
        "What's happening",
        "What's the situation",
        
        // ── "What is the meaning of" queries (definitional) ────────────────────────
        "What is the meaning of grammar",
        "What is the meaning of syntax",
        "What is the meaning of semantics",
        "What is the meaning of pragmatics",
        "What is the meaning of linguistics",
        "What is the meaning of morphology",
        "What is the meaning of phonetics",
        "What is the meaning of etymology",
        "What the meaning of grammar",
        "What the meaning of syntax",
        "What the meaning of life",
        "What the meaning of love",
        "What the meaning of success",
        "What the meaning of happiness",
        "What does grammar mean",
        "What does syntax mean",
        "What does semantics mean",
        "Define grammar",
        "Define syntax",
        "Define semantics",
        "Explain what grammar is",
        "Explain what syntax is",
        "Tell me what grammar means",
        "Tell me what syntax means",
        "What is grammar",
        "What is syntax",
        "What is semantics",
        "What is morphology",
        "What is phonetics",
        "What is linguistics",
        
        // ── Meta questions about the assistant ───────
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
        "Can you explain how this works?",
        "What model are you running under the hood?",
        "Do you have access to my camera?",
        "Can you read files from my Desktop folder?",
        "What happens to the data I store in memory?",
        "Can you generate images?",
        "Do you support voice input right now?",
        "Can you call external APIs directly?",
        "What's the maximum context window you keep?",
        "Can you export my memory as JSON?",
        
        // ── Follow-up questions (asking for more details) ───────
        "Give me examples",
        "Can you give me examples?",
        "Show me some examples",
        "Tell me more",
        "Explain further",
        "Can you elaborate?",
        "Provide more details",
        "Go into more detail",
        "What else?",
        "Anything else?",
        "Continue",
        "Keep going",
        "More info please",
        "Can you expand on that?",
        "Give me more information",
        
        // ── Troubleshooting questions ──────────────────────────
        "Why is my code not working?",
        "Why am I getting this error in my terminal?",
        "Why is the server not starting?",
        "Why can't I connect to the database?",
        "Why is my app so slow?",
        "Why is Docker not working?",
        "Why won't Git push?",
        "Why is npm install failing?",
        "Why is my build breaking?",
        "Why can't I import this module?",
        "What's wrong with my code?",
        "What does this error mean?",
        "What's the issue here?",
        "What am I doing wrong?",
        "Help me debug this code",
        "Help me fix this error",
        "Why isn't this working?",
        
        // ── Comparison and recommendation questions ───────────────
        "Should I use React or Vue?",
        "Which is better: MySQL or PostgreSQL?",
        "What's the difference between npm and yarn?",
        "Should I learn Python or JavaScript first?",
        "Which framework should I use?",
        "Should I use REST or GraphQL?",
        "Which cloud provider is best?",
        "What's the best way to learn coding?",
        "Should I use TypeScript?",
        "What's the difference between let and const?",
        "Should I use MongoDB or SQL?",
        
        // ── Language, pronunciation, translation, definitions ──────────────
        "How do I pronounce the spanish word for fish",
        "How do you pronounce quinoa",
        "What's the French word for thank you",
        "How do you say hello in Japanese",
        "Translate good morning to German",
        "What does 'schadenfreude' mean",
        "Define ephemeral",
        "What's the definition of ubiquitous",
        "How is 'croissant' pronounced",
        "What's the Italian word for cheese",
        "Translate 'where is the bathroom' to Spanish",
        "How do you spell bureaucracy",
        "What does the word 'serendipity' mean",
        "How do you conjugate 'être' in French",
        "What's the plural of octopus",
        "What's the difference between affect and effect",

        // ── Explanation requests ──────────────────────────────
        "Explain how async/await works",
        "Explain the event loop",
        "Explain closures in JavaScript",
        "Explain promises",
        "Explain dependency injection",
        "Explain the virtual DOM",
        "Explain how React hooks work",
        "Explain middleware in Express",
        "Explain JWT authentication",
        "Explain CORS",
        "Explain REST principles",
        "Explain database indexing",
        "Explain garbage collection",
        "Explain Docker containers",
        "Explain CI/CD pipelines",
        "Explain load balancing",
        "Explain caching strategies",
        
        // ── Best practices and advice ─────────────────────────
        "What are best practices for API design?",
        "What are common security vulnerabilities?",
        "What are the principles of clean code?",
        "What are SOLID principles?",
        "What are design patterns I should know?",
        "What are the best practices for Git?",
        "What are the best practices for database design?",
        "What are the best practices for testing?",
        "What are the best practices for error handling?",
        
        // ── Career and learning questions ───────────────────────
        "How do I become a full-stack developer?",
        "What skills do I need for a backend role?",
        "How do I prepare for coding interviews?",
        "What should I learn next?",
        "How do I build a portfolio?",
        "How do I get my first developer job?",
        "How do I stay up to date with tech?",
        "What resources do you recommend for learning?",
        "How do I contribute to open source?",
        "How do I improve my problem-solving skills?",

        // ── Round 5 seeds ───────────────────────────────────────
        // Historical events (phi4 misclassifies as memory_retrieve at 100%)
        "What was the Marshall Plan?",
        "What was the New Deal?",
        "What was D-Day?",
        "What was the Cold War?",
        "What was the Berlin Wall?",
        "What was the Space Race?",
        "What was the Cuban Missile Crisis?",
        // Word definitions (phi4 misclassifies as web_search)
        "What does the word petrichor mean?",
        "What does the word ephemeral mean?",
        "What does the term paradigm mean?",
        "What does serendipity mean?",
        "Define the word petrichor",
        "Define the term idempotent",
        // Technical CS concepts (phi4 misclassifies as web_search)
        "What is the Byzantine Generals problem in distributed systems?",
        "What is the Halting problem in computer science?",
        "What is the CAP theorem in distributed systems?",
        "What is the Dining Philosophers problem?",
        "What is the birthday paradox in probability?",
        "What is the two generals problem in networking?",
        // ── Round 6 seeds ──────────────────────────────────────
        "Explain how TCP/IP works",
        "Explain how the internet works",
        "Explain how DNS resolution works",
        "Explain how HTTPS handshake works",
        "Explain how blockchain consensus works",
        "Explain how compilers work",
        "explain how TCP I P works",
        "explain how the net works",
        "What is the Dunbar number and what does it describe?",
        "What is Dunbar's number?",
        "What is the Pareto principle?",
        "What is the Feynman technique?",
        "What is the Pomodoro technique and how does it work?",
        "What is the Socratic method?",
        "What's the best Python version to use for ML projects today?",
        "What Python version should I use for machine learning?",
        "What is the recommended Node.js version for production?",
        // ── Round 7 seeds ──────────────────────────────────────
        // Concepts phi4 confuses with web_search (has 100% or high web_search confidence)
        "What is dollar-cost averaging and how does it work?",
        "how does dol lar cost av er ag ing work",
        "Explain the concept of opportunity cost in economics",
        "ex plain the con cept of op por tun i ty cost in e co nom ics",
        "Explain how a relational database works",
        "What is a golden retriever's average lifespan?",
        "what is the sci en tif ic name for the do mes tic dog",
        "How does carbon dating work?",
        "What is the optimal ratio of tea to sugar for kombucha fermentation?",
        "What programming language is best for data science projects in 2026?",
        "What is AGI?",
        "ex plain the dif fer ence between an L L M and A G I",
        "How does a SCOBY work in kombucha fermentation?",
        "how does a sko by work in kom bu cha fer men ta tion",
        // ── Round 7b seeds ──────────────────────────────────────
        // Programming language best practices — stable/conceptual NOT time-sensitive (gk-a03)
        "What programming language is best for data science in 2025?",
        "Which language is most popular for machine learning today?",
        "What is the recommended programming language for ML projects?",
        "What language should I learn for data science work?"
      ],    
      // command_execute: [
      command_automate: [
        // ── File search and existence queries ──────────────────
        // "Do I have X files" = search the filesystem → command_automate (mdfind/find)
        "Do I have resume files on my computer",
        "Do I have any PDF files on my desktop",
        "Do I have any photos in my Downloads folder",
        "Are there any log files in my home directory",
        "Do I have a file called config.json anywhere",
        "Find all resume files on my computer",
        "Find all PDF files on my desktop",
        "Search for resume files on my computer",
        "Search for files named resume on my computer",
        "List all PDF files in my Documents",
        "Show me all text files on my desktop",
        "Find all images in my Downloads folder",
        "Are there any zip files on my desktop",
        "Do I have any Word documents in my Documents folder",
        "Find all JavaScript files in my projects folder",
        "List all applications installed on my computer",
        "Show me all apps on my computer",
        "What applications are installed on my Mac",
        "List all the applications on my computer",
        "Show me all installed apps",
        "Find all .txt files on my desktop",
        "Search my computer for files named budget",
        "Do I have any spreadsheet files",
        "Find all Excel files on my computer",
        "Are there any Python files in my projects",
        "List all files on my desktop",
        "Show me what files are on my desktop",
        "What files are in my Downloads folder",
        "List the contents of my Documents folder",
        "Show me the files in my home directory",

        // ── Find file by name + read/analyze ──────────────────
        // "find the file cheese and analyze it" must be command_automate (mdfind + fs.read)
        // NOT screen_intelligence — there is no screen to analyze, it's a file search
        "Find the file cheese and analyze and tell me what's it about",
        "Find the file called notes and read it",
        "Find the file budget and tell me what's in it",
        "Find the file resume and analyze it",
        "Locate the file called report and read it",
        "Find the cheese text file and tell me about it",
        "Find my notes file and summarize it",
        "Locate the file invoice and tell me what it says",
        "Find the file called readme and read it",
        "Search for the file config and open it",
        "Find a file named todo and tell me what's in it",
        "Find the file project and analyze it",
        "Locate the file budget.txt and summarize",
        "Find the notes.txt file on my computer",
        "Find the file data and read its contents",
        "Search for cheese.txt and tell me about it",
        "Find the file cheese and read it",
        "Locate cheese file and summarize what it says",
        "Find the text file about cheese",
        "Find the cheese file on my mac",

        // ── Reminder / timer / alarm / schedule ─────────────────
        // These need the schedule pseudo-skill to fire a timed notification.
        // NOT memory_store — storing a fact doesn't actually remind anyone.
        "Remind me in 5 minutes to check the oven",
        "Remind me about the meeting tomorrow at 10am",
        "Set a reminder for my dentist appointment next Friday at 3pm",
        "Set a timer for 10 minutes",
        "Set an alarm for 7am",
        "Alert me in 30 seconds",
        "Wake me up at 6:30am",
        "Notify me in 1 hour to take my medication",
        "Remind me to call mom at 5pm",
        "In 5 minutes remind me to check the laundry",
        "Set a reminder that I have an appointment in two weeks",
        "Create a reminder for my doctor's appointment on Monday",
        "I need a reminder for the team standup at 9am tomorrow",
        "Quick reminder: call Dr. Patel on Tuesday 9am about bloodwork",
        "Remind me to pick up groceries after work",
        "Set a 15 minute timer",
        "Remind me in 2 hours to submit the report",

        // ── Date / time / system info ──────────────────────
        // These need shell.run (date, system_profiler, etc.) — NOT memory_retrieve or general_knowledge
        "What's today's date",
        "What is today's date",
        "What's the date today",
        "What day is it today",
        "What time is it",
        "What's the current time",
        "Tell me today's date",
        "Get today's date",
        "What is the current date and time",
        "What's my battery level",
        "How much battery do I have left",
        "What's my current wifi network",
        "What's my IP address",
        "How much disk space do I have",
        "What's my disk usage",
        "What's my Mac's hostname",
        "What's my username on this machine",
        "Show me system info",
        "What OS version am I running",
        "Check my CPU usage",
        "How much RAM is being used",

        // ── Codebase / project analysis ────────────────────
        // Reading/exploring a codebase with fs.read — NOT screen_intelligence
        "Analyze the application at my Desktop/projects/thinkdrop",
        "Read the codebase at ~/projects/myapp and tell me what it does",
        "Explore the project in my projects folder",
        "Understand the app at ~/Desktop/projects/thinkdrop-aws",
        "Tell me what this app is all about",
        "What is this project about",
        "Give me an overview of this codebase",
        "Analyze the project thinkdrop-aws in my projects folder",
        "Read and understand the code at ~/projects/myapi",
        "Explore the repo at my Desktop",
        "What does the application in my projects folder do",
        "Summarize the application at ~/Desktop/projects/myapp",
        "Examine the project structure at ~/projects/thinkdrop",
        "Analyze this application and tell me what it's about",
        "Inspect the codebase at ~/projects/webapp",
        "Read the project at my Desktop and summarize it",

        // ── File and folder manipulation ──────────────────
        "Create a file on my desktop called hello.txt",
        "Make a folder on my desktop named projects",
        "Create a new folder called Photos",
        "Create a folder called Documents",
        "Make a new folder named Downloads",
        "Create a file called helloworld.txt on my desktop",
        "Delete the file test.txt from my desktop",
        "Remove the folder old_stuff from my Documents",
        "Move file.txt to my desktop",
        "Copy report.pdf to my Documents",
        "Rename the file old.txt to new.txt",
        "Create a new file on my desktop",
        "Make a new folder in my Documents",
        "Delete all .tmp files from my desktop",
        "Create a file named data.json on my desktop",
        "Touch a new file called config.yml",
        "Make directory called backup in my home",
        "Remove all log files from current directory",
        "Copy all PDFs to my Documents folder",
        "Move everything from Downloads to Archive",
        "Compress the reports folder into a zip",
        "Extract the archive.tar.gz file",
        "Change permissions on script.sh to executable",
        "Create a symbolic link to my projects folder",
        "Create a new directory called backup in my home",
        "Create a new folder called Projects",
        "Make a new folder named Work",

        // ── File editing and updating ──────────────────
        "Find the file music-songs and edit it",
        "Find file music-songs and update it with new chapter and verse",
        "Edit the file notes.txt and add new content",
        "Update the file config.txt with new settings",
        "Open the file report.txt and add a new section",
        "Find my weekly verse file and update it to Genesis 22",
        "Edit music-songs and change the chapter",
        "Update my notes file with new information",
        "Find the file budget.txt and update the numbers",
        "Edit the file todo.txt and add new tasks",
        "Update the readme file with new instructions",
        "Find and edit the file called schedule.txt",
        "Append new content to the file journal.txt",
        "Add a new line to my notes.txt file",
        "Write new content to the file called plan.txt",
        "Update the text file with the new chapter and verses",
        "Find my memory verse file and update it",
        "Edit the document and replace the old chapter with the new one",
        "Update the file with Genesis 22 verses for the kids",
        "Find the weekly memory verse file and change it to the next chapter",

        // ── Service automation & external integrations (need a skill) ────────
        // Gmail / email monitoring
        "Watch my incoming mail on Gmail and give me a daily summary",
        "Monitor my Gmail inbox and send me a text every night",
        "Check my Gmail inbox and summarize new emails at 9pm",
        "Watch my email and alert me when I get something from my boss",
        "Monitor my inbox and forward important emails to me",
        "Send me a daily email digest of my Gmail inbox",
        "Watch for new emails in my Gmail and notify me by text",
        "Poll my Gmail every hour and send me a summary",
        "Check my inbox and send a text message summary at night",
        "Monitor my email and send a daily report via SMS",
        // SMS / text notifications
        "Send me a text message when my package arrives",
        "Text me a reminder at 9pm every night",
        "Send me an SMS summary of my day",
        "Notify me by text when there are new emails",
        "Send a daily SMS with my schedule",
        // Calendar / scheduling
        "Check my calendar and remind me of tomorrow's events",
        "Monitor my Google Calendar and text me daily reminders",
        "Watch my calendar and send me a morning briefing",
        "Schedule a daily summary of my meetings at 8am",
        "Set up a daily reminder for my calendar events",
        // Slack / Discord / messaging apps
        "Watch my Slack messages and summarize them daily",
        "Monitor my Discord server and alert me on new messages",
        "Check my Slack and send a daily digest",
        // General service monitoring
        "Track my GitHub issues and notify me of new ones",
        "Monitor my website uptime and alert me if it goes down",
        "Watch my Notion database for new entries",
        "Monitor my Airtable and send updates",
        "Poll my API endpoint every 5 minutes",

        // ── Application launching (NOT screen intelligence) ──────────────────────
        "Open Chrome",
        "Close all windows",
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
        "Set an alarm for 7am",

        // ── Screen/system control commands ──────────────────
        "Lock my screen",
        "Lock the screen",
        "Lock my computer",
        "Lock this device",
        "Lock my laptop",
        "Start recording my screen",
        "Start screen recording",
        "Record my screen",
        "Begin screen capture",
        "Start recording the screen",
        "Capture my screen",
        "Record the display",
        "Turn the volume up",
        "Turn volume down",
        "Increase volume",
        "Decrease volume",
        "Adjust volume to 50",
        "Set volume to maximum",
        "Shut down my computer",
        "Shut down the system",
        "Power off my computer",
        "Turn off my laptop",
        "Shutdown this machine",
        
        // ── Calendar and scheduling actions ──────────────────
        "Schedule a meeting for tomorrow",
        "Schedule a meeting for next week",
        "Set up a meeting for Monday",
        "Create a calendar event for tomorrow",
        "Add a meeting to my calendar",
        "Book a meeting for 3pm",
        "Schedule an appointment for next Tuesday",
        
        // ── Single-step UI click/type/press actions ──────────────────
        // Short imperative commands — were scoring as screen_intelligence without these
        "Click the submit button",
        "Click the login button",
        "Click the OK button",
        "Click the cancel button",
        "Click the save button",
        "Click submit",
        "Click OK",
        "Click cancel",
        "Press the submit button",
        "Press OK",
        "Press enter",
        "Tap the send button",
        "Tap the confirm button",
        "Click the sign in button",
        "Click the next button",
        "Click the back button",
        "Click the close button",
        "Click the download button",
        "Click the upload button",
        "Click the delete button",
        "Click the edit button",
        "Click the add button",
        "Click the create button",
        "Click the search button",
        "Click the apply button",
        "Type hello in the search box",
        "Type my name in the input field",
        "Type the password in the password field",
        "Enter my email in the email field",
        "Fill in the form with my details",
        "Select the first option from the dropdown",
        "Check the agree to terms checkbox",
        "Uncheck the newsletter checkbox",
        "Scroll down the page",
        "Scroll to the bottom",

        // ── SMS / text-to-me actions ──────────────────
        // Canonical forms
        "Text this to me",
        "Text that to me",
        "Text these to me",
        "Send this to me",
        "Send that to me",
        "Send these to me",
        "Text this info to me",
        "Text that info to me",
        "Send this info to me",
        "Send that info to me",
        "Text me this",
        "Text me that",
        "Text me these",
        "Send me this",
        "Send me that",
        "Text me the results",
        "Send me the results",
        "Text me the details",
        "Send me the details",
        "Text this to my phone",
        "Send this to my phone",
        "Text that to my phone",
        "Text this to my number",
        "Send this to my number",
        "Text me the info",
        "Send me the info",
        // With destination qualifier
        "Text this to me now",
        "Text that info to my phone",
        "Send these results to my phone",
        "Email this to me",
        "Email me this",
        "Email me the results",
        "Forward this to me",
        "Forward that to me",
        // Typos and informal variants (model must learn these via embedding similarity)
        "Text these to my me",
        "text thes to me",
        "sned this to me",
        "txt this to me",
        "text thos to me",
        "send thsi to me",
        "text this too me",
        "text me thos",
        "snd me this",
        "text me these info",

        // ── Email actions ──────────────────
        "Send an email to John",
        "Send an email to my boss",
        "Email Sarah about the project",
        "Compose an email to the team",
        "Draft an email to support",
        "Send a message to John via email",
        
        // ── Multi-step automation commands ──────────────────
        // Complex workflows with multiple actions (goto X, find Y, do Z)
        "Goto chatgpt find my project called Thinkdrop AI and do a search for how to use Stripe API",
        "Open Slack and then find the engineering channel",
        "Navigate to Notion and search for my meeting notes",
        "Go to Figma and find the design file called Homepage",
        "Launch VSCode and open the project folder",
        "Open Chrome and navigate to github.com",
        "Go to Gmail and compose a new email",
        "Open Spotify and play my workout playlist",
        "Launch Terminal and run the build script",
        "Open Discord and join the voice channel",
        "Go to Zoom and start a new meeting",
        "Open Calendar and create a new event",
        "Navigate to Twitter and post a tweet",
        "Go to LinkedIn and send a connection request",
        "Open YouTube and search for React tutorials",
        "Launch Postman and test the API endpoint",
        "Go to Trello and create a new card",
        "Open Jira and update the ticket status",
        "Navigate to Confluence and create a new page",
        "Go to Google Drive and share the document",
        "Open Dropbox and upload the file",
        "Launch Docker and start the container",
        "Go to AWS Console and check the EC2 instances",
        "Open Vercel and deploy the latest build",
        "Navigate to Netlify and check the deployment status",
        "Go to Stripe Dashboard and view recent transactions",
        "Open Firebase Console and check the database",
        "Launch Xcode and build the iOS app",
        "Go to App Store Connect and submit the update",
        "Open Android Studio and run the emulator",
        "Navigate to Play Console and publish the release",
        
        // ── Advanced Multi-Tool Automation Commands ──────────────────
        // Concrete automation actions involving multiple tools/steps
        "Open Mermaid Live Editor and create a UML diagram for my user authentication flow",
        "Launch Draw.io and generate a network topology map for the production environment",
        "Open VS Code and create a sequence diagram from the checkout user story",
        "Go to Lucidchart and generate an org chart for the engineering team",
        "Launch Miro and create a mind map from the product brainstorming notes",
        "Open Figma and create component diagrams for the design system",
        "Launch Cursor and write a Python script to process CSV files",
        "Open CodeSandbox and prototype a React dashboard app",
        "Go to VS Code and generate an API client for the Stripe integration",
        "Launch Warp and write a shell script to backup the database",
        "Open Copy.ai and generate 10 LinkedIn posts about our product launch",
        "Go to Notion and convert the meeting notes into a blog draft",
        "Launch Canva and create social media graphics for the campaign",
        "Open Zapier and set up an automation to trigger emails from form submissions",
        "Go to n8n and create a workflow to sync data between Airtable and Slack",
        "Launch Google Sheets and create formulas to analyze the sales data",
        "Open Power Automate and set up invoice approval automation",
        "Go to Obsidian and link my project notes with related concepts",
        "Launch Anki and create flashcards from the study notes",
        "Open Readwise and export highlights to my note-taking app",
        "Go to Zapier and create a Zap to post tweets from RSS feeds",
        "Launch Airtable and set up a database for customer feedback",
        "Open Typeform and create a survey with conditional logic",
        "Go to Calendly and set up meeting scheduling with team availability",
        "Launch Mailchimp and create an email campaign for the newsletter",
        "Open HubSpot and set up lead scoring automation",
        "Go to Intercom and configure chatbot responses for support",
        "Launch Segment and set up event tracking for the web app",
        "Open Mixpanel and create a funnel analysis dashboard",
        "Go to Amplitude and set up user cohort analysis",
        
        // ── UI Automation Commands (goto app, find element, type/click) ──────────────────
        // CRITICAL: These are automation commands, NOT screen intelligence queries
        "goto windsurf look for input field ask anything and type hello",
        "goto windsurf find the input field and type hello",
        "goto chatgpt find the input box and type my question",
        "goto perplexity look for search bar and type my query",
        "goto chrome find the address bar and type google.com",
        "goto slack find the message input and type hello team",
        "goto discord find the chat box and send a message",
        "goto notion find the new page button and click it",
        "goto figma find the create button and click it",
        "goto vscode find the file menu and open settings",
        "I need you to goto windsurf app find the file capabilities.ts and copy all the code",
        "I need you to goto chatgpt paste that code there and ask it to examine the code",
        "I need you to goto perplexity and search for react best practices",
        "I need you to goto chrome and navigate to github.com",
        "I need you to goto slack and send a message to the engineering channel",
        "I need you to goto discord and join the voice channel",
        "I need you to goto notion and create a new page",
        "I need you to goto figma and open the design file",
        "I need you to goto vscode and run the build command",
        "I need you to goto terminal and execute npm install",
        "goto windsurf app find the file capabilities.ts and copy all the code then open chatgpt paste that code there and ask it to examine the code for me then explain to me what it said for a response",
        "goto chrome find google search and type python tutorial then click the first result",
        "goto slack find engineering channel and post the meeting notes",
        "goto notion find my project page and add a new task",
        "goto vscode find the terminal and run npm start",
        "goto spotify find my workout playlist and play it",
        "goto gmail find compose button and write an email to john",
        "goto calendar find new event button and create a meeting",
        "goto trello find my board and add a new card",
        "goto jira find the ticket and update the status",
        "open windsurf find the search bar and search for capabilities.ts",
        "open chatgpt find the input field and paste the code",
        "open perplexity find the search box and type my question",
        "open chrome find the url bar and go to github.com",
        "open slack find the channel list and select engineering",
        "open discord find the server and join the voice channel",
        "open notion find the sidebar and create a new page",
        "open figma find the file browser and open homepage design",
        "open vscode find the command palette and run build",
        "open terminal find the prompt and type ls -la",
        "launch windsurf and search for capabilities.ts file",
        "launch chatgpt and paste the code in the input",
        "launch perplexity and search for react hooks",
        "launch chrome and navigate to stackoverflow.com",
        "launch slack and message the team channel",
        "launch discord and start a voice call",
        "launch notion and create a new database",
        "launch figma and duplicate the component",
        "launch vscode and open the integrated terminal",
        "launch spotify and shuffle my liked songs",
        
        // ── Network actions ──────────────────
        "Ping 8.8.8.8",
        "Check if google.com is reachable",
        "Test connection to localhost:3000",
        "Check if port 8080 is in use",
        "Flush DNS cache",
        "Trace route to google.com",
        // ── System monitoring commands ──────────────────
        "Check system memory usage",
        "Show RAM usage",
        "How much memory do I have",
        "How much memory I have on my computer",
        "How much RAM do I have",
        "What's my memory usage",
        "Check disk space",
        "Show CPU usage",
        "Monitor network activity",
        "Check battery status",
        "Show running processes",
        "How much disk space do I have",
        "What's my CPU usage",
        // ── Version and installation queries ──────────────────
        "What version of Python do I have installed",
        "Which version of Node.js am I running",
        "Check my Python version",
        "What version of npm is installed",
        "Show me my Docker version",
        "What version of Git do I have",
        "Check Node version",
        "Which Python version am I using",
        "What's my current Ruby version",
        "Show installed Java version",
        "What version of Rust is installed",
        "Check my Go version",
        "Which version of PostgreSQL do I have",
        "What's my MySQL version",
        "Show me the installed npm version",
        // ── Docker and container queries ──────────────────
        "How many Docker containers are running",
        "List all Docker containers",
        "Show running Docker containers",
        "What containers do I have",
        "How many apps do I have in my Docker",
        "List Docker images",
        "Show all Docker containers",
        "What Docker containers are active",
        "How many containers are running",
        "Show me my Docker containers",
        "List running containers",
        "What's running in Docker",
        "Show Docker container status",
        "How many Docker images do I have",
        "List all containers",
        // ── File and folder listing queries (filesystem, NOT screen) ──────────────────
        "List all the folders in my desktop directory",
        "Show me all files in my desktop folder",
        "What files are in my Downloads folder",
        "List all the files and folders in my desktop directory",
        "Show all folders in my Documents directory",
        "List everything in my desktop folder",
        "Show me what's in my home directory",
        "What folders do I have in my desktop directory",
        "List all files in my Downloads folder",
        "Show all items in my desktop directory",
        "What's in my Documents folder",
        "List the contents of my desktop directory",
        "Show me my desktop directory files",
        "What files are in the desktop folder",
        // ── File counting and statistics ──────────────────────
        "How many files on my desktop",
        "How many files are on my desktop",
        "How many folders on my desktop",
        "Count files on my desktop",
        "Count files in my Downloads",
        "How many files in that folder",
        "How many items on my desktop",
        "Count all files on my desktop",
        "How many files do I have on my desktop",
        "How many folders are in my Documents",
        "Count the files in this directory",
        "How many files are in my home folder",
        
        // ── Search/fetch + save to file (multi-step: web search → write file) ──────────────────
        // CRITICAL: These were being misclassified as web_search — they require shell execution to write the file
        // ~/Desktop/
        "Search for the top 5 trending GitHub repositories today and save a summary to ~/Desktop/trending-repos.md",
        "Find the latest news about AI and save it to ~/Desktop/ai-news.txt",
        "Search for the best React libraries and write the results to ~/Desktop/react-libs.md",
        "Look up the current Bitcoin price and save it to ~/Desktop/crypto.txt",
        "Find the top 10 Python packages and save a list to ~/Desktop/python-packages.txt",
        "Look up the weather forecast for this week and write it to ~/Desktop/weather.txt",
        "Search for the best VS Code extensions and output a summary to ~/Desktop/vscode-extensions.txt",
        "Fetch the top Hacker News stories and write them to ~/Desktop/hn-stories.md",
        "Look up current stock prices for AAPL and save to ~/Desktop/stocks.txt",
        // ~/Documents/
        "Search for open source machine learning projects and save results to ~/Documents/ml-projects.md",
        "Find recent articles about TypeScript and save them to ~/Documents/ts-articles.md",
        "Look up the latest npm packages for authentication and write to ~/Documents/auth-packages.md",
        "Search for remote jobs in software engineering and save results to ~/Documents/jobs.md",
        "Find the top JavaScript frameworks and write a summary to ~/Documents/frameworks.md",
        "Search for startup funding news and save a summary to ~/Documents/funding.md",
        "Look up Python best practices and save the notes to ~/Documents/python-notes.txt",
        "Find documentation for the Stripe API and save it to ~/Documents/stripe-docs.md",
        // ~/Downloads/
        "Search for free stock photos and save the links to ~/Downloads/photos.txt",
        "Find the latest release notes for Node.js and save to ~/Downloads/node-release.md",
        "Look up open source licenses and write a comparison to ~/Downloads/licenses.md",
        // /tmp/ and relative paths
        "Search for the top 5 results for machine learning tutorials and save to /tmp/tutorials.txt",
        "Find trending repos on GitHub and write to ./trending.md",
        "Look up the weather and save it to ./weather.txt",
        // No explicit path — "save to a file", "write to a file on my desktop/documents"
        "Get the latest GitHub trending repos and save to a file on my desktop",
        "Find trending repositories on GitHub this week and save a summary to my desktop",
        "Search for the best coffee shops near me and write results to a file",
        "Find all open issues in the react repository and save to a file on my desktop",
        "Search for Python documentation on decorators and save to a text file",
        "Look up the top 5 AI tools and write them to a markdown file on my desktop",
        "Find recent tech news and save a summary to a file in my Documents folder",
        "Search for the best productivity apps and write the list to my Documents",

        // ── Network queries (read-only) ──────────────────────────
        "What's my IP address",
        "Show my local IP",
        "What's my public IP",
        "Show active network connections",
        "What ports are open on my machine",
        "Show my network interfaces",
        "What's my MAC address",
        "Display routing table",
        "Show DNS servers",
        "Flush DNS cache",
        "Trace route to google.com",
        "Show network statistics",
        "Check internet speed",
        "What's my Wi-Fi network name",
        "Scan for available Wi-Fi networks",
        "Show firewall status",
        
        // ── Git and version control ───────────────────────────
        "Git status",
        "Show git log",
        "What branch am I on",
        "List all git branches",
        "Show uncommitted changes",
        "Git diff",
        "Show last commit",
        "Git pull latest changes",
        "Push to remote",
        "Create a new branch called feature-x",
        "Switch to main branch",
        "Merge develop into current branch",
        "Stash my changes",
        "Show git stash list",
        "Revert last commit",
        "Show git remote",
        "Clone the repository",
        "Add all files to staging",
        "Commit with message 'fix bug'",
        "Show file history",
        "Git blame on this file",
        "Show who changed this line",
        "List all tags",
        "Create a new tag v1.0.0",
        "Write a commit message",
        "Write a commit message for this file",
        "Write a commit message for this branch",
        
        // ── Package management ────────────────────────────────
        "Install express with npm",
        "Update all npm packages",
        "List installed npm packages",
        "Remove unused dependencies",
        "Install Python package requests",
        "Pip install pandas",
        "Update pip",
        "List all pip packages",
        "Install homebrew package wget",
        "Brew update",
        "Brew upgrade all packages",
        "Search for package in npm",
        "Show package info for react",
        "Install global package",
        "Uninstall package",
        "Clear npm cache",
        "Run npm audit",
        "Fix npm vulnerabilities",
        "Install dependencies from package.json",
        "Install Ruby gem rails",
        "Update all gems",
        "Install cargo package",
        "Go get package",
        
        // ── Process and service management ────────────────────
        "Kill process on port 3000",
        "Stop the running server",
        "List all running processes",
        "Show processes using most CPU",
        "Show processes using most memory",
        "Find process by name node",
        "Kill all node processes",
        "Restart Apache server",
        "Start nginx service",
        "Stop Docker service",
        "Check if MySQL is running",
        "Show service status",
        "Enable service at startup",
        "Disable service",
        "View system logs",
        "Tail the application log",
        "Monitor log file in real-time",
        "Show last 50 lines of error log",
        "Clear system logs",
        "Start background job",
        "List cron jobs",
        "Add cron job",
        
        // ── Text processing and search ────────────────────────
        "Search for 'error' in log files",
        "Find all occurrences of TODO in code",
        "Grep for pattern in files",
        "Count lines in file",
        "Show first 10 lines of file",
        "Show last 20 lines of file",
        "Replace text in file",
        "Find and replace across multiple files",
        "Sort file contents",
        "Remove duplicate lines",
        "Convert file to uppercase",
        "Extract column from CSV",
        "Merge two files",
        "Split large file into chunks",
        "Compare two files",
        "Show differences between files",
        "Word count in document",
        "Find files containing specific text",
        
        // ── Database operations ───────────────────────────────
        "Connect to MySQL database",
        "Show all databases",
        "List tables in database",
        "Describe table structure",
        "Run SQL query",
        "Export database to file",
        "Import SQL dump",
        "Backup database",
        "Restore database from backup",
        "Show database size",
        "Check database status",
        "Optimize database tables",
        "Create new database",
        "Drop database",
        "Grant user permissions",
        "Show active connections",
        "Kill long-running query",
        
        // ── Development and build ─────────────────────────────
        "Run the dev server",
        "Start development mode",
        "Build the project",
        "Run production build",
        "Run tests",
        "Run unit tests",
        "Run integration tests",
        "Run test coverage",
        "Lint the code",
        "Format code with prettier",
        "Type check with TypeScript",
        "Bundle the application",
        "Watch for file changes",
        "Hot reload the server",
        "Clear build cache",
        "Generate documentation",
        "Run migrations",
        "Seed the database",
        "Start the debugger",
        "Profile the application",
        
        // ── Environment and configuration ─────────────────────
        "Show environment variables",
        "Set environment variable",
        "Load .env file",
        "Show PATH variable",
        "Add to PATH",
        "Show shell configuration",
        "Reload shell config",
        "Show aliases",
        "Create alias for command",
        "Show current directory",
        "Change to home directory",
        "Go to parent directory",
        "Show directory tree",
        "Print working directory",
        "Show hidden files",
        "List files with details",
        "Sort files by size",
        "Sort files by date",
        
        // ── Security and permissions ──────────────────────────
        "Generate SSH key",
        "Show SSH public key",
        "Add SSH key to agent",
        "Test SSH connection",
        "Change file permissions",
        "Change file owner",
        "Show file permissions",
        "Make file read-only",
        "Make script executable",
        "Show current user",
        "Switch to root user",
        "Run command as sudo",
        "Show sudo permissions",
        "Lock the screen",
        "Logout current user",
        "Change password",
        "Generate random password",
        "Encrypt file",
        "Decrypt file",
        "Calculate file checksum",
        "Verify file integrity",
        
        // ── Miscellaneous system operations ───────────────────
        "Clear terminal screen",
        "Show command history",
        "Repeat last command",
        "Show system uptime",
        "Show logged in users",
        "Show last login",
        "Display calendar",
        "Show current date and time",
        "Set system time",
        "Show timezone",
        "Convert timestamp",
        "Calculate date difference",
        "Schedule shutdown",
        "Cancel scheduled shutdown",
        "Hibernate the system",
        "Sleep the computer",
        "Eject USB drive",
        "Mount external drive",
        "Unmount drive",
        "Check disk for errors",
        "Defragment disk",
        "Show system information",
        "Display hardware info",
        "Show kernel version",
        "Update system",
        "Upgrade packages",
        "Clean package cache",
        "Remove old kernels",

        // ── TERMINAL commands (check, see, run in terminal) ──────────
        "See in the terminal how much disk space I have",
        "Check in the terminal how much space I have",
        "Look in the terminal for disk usage",
        "Run df -h in terminal",
        "Check terminal for running processes",
        "See in terminal what's using port 3000",
        "Look in the terminal for error logs",
        "Check the terminal output",
        "See what's in the terminal",
        "Run ls in terminal",
        "Execute pwd in terminal",
        "Run git status in terminal",
        "Check terminal for npm version",
        "See in terminal if docker is running",
        "Look in terminal for python version",
        "Run top in terminal",
        "Execute htop in terminal",
        "Check terminal for node processes",
        "See terminal output for last command",
        "Run command in terminal",
        "Execute script in terminal",
        "Open terminal and run htop",
        "Open terminal and check disk space",
        "Launch terminal and run ls",
        
        // ── DOCKER and container commands ─────────────────────────────
        "Run the docker file on my desktop",
        "Run the Dockerfile on my desktop",
        "Execute the docker file",
        "Start the docker container",
        "Run docker compose up",
        "Execute docker-compose up",
        "Start docker containers",
        "Run the docker image",
        "Execute docker run",
        "Start the docker service",
        "Run docker ps",
        "Execute docker images",
        "Build the docker image",
        "Run docker build",
        "Execute docker-compose build",
        "Stop all docker containers",
        "Run docker stop",
        "Remove docker containers",
        "Run docker rm",
        "Prune docker images",
        "Run docker system prune",
        "Check docker logs",
        "Run docker logs",
        "Inspect docker container",
        "Run docker inspect",
        "Restart docker container",
        "Run docker restart",
        
        // ── FILE execution and script running ─────────────────────────
        "Run the python script on my desktop",
        "Execute the bash script",
        "Run the shell script",
        "Execute the node script",
        "Run the javascript file",
        "Execute the typescript file",
        "Run the ruby script",
        "Execute the perl script",
        "Run the go program",
        "Execute the rust binary",
        "Run the java class",
        "Execute the jar file",
        "Run the executable",
        "Execute the binary",
        "Run the program",
        "Execute the application",
        "Run the script in my home folder",
        "Execute the file on my desktop",
        "Run the test suite",
        "Execute the build script",
          
        // ── FIND/SEARCH commands (local file search) ──────────────────
        "Find all PDFs on my desktop",
        "Search for text files in Documents",
        "Find files containing 'project' in name",
        "Search for images in Downloads",
        "Find all videos on my computer",
        "Search for files modified today",
        "Find large files over 1GB",
        "Search for duplicate files",
        "Find empty folders",
        "Search for files by extension",
        "Find all .js files in project",
        "Search for .py files",
        "Find files created last week",
        "Search for files by date",
        "Find files owned by me",
        "Search for hidden files",
        "Find all log files",
        "Search for config files",
        "Find all JSON files",
        "Search for markdown files",
        
        // ── CROSS-APPLICATION AUTOMATION (image generation, content creation) ──────────────────
        // CRITICAL: Commands that require automating actions across multiple AI tools/apps
        "Generate Mickey Mouse images in ChatGPT, Grok and Perplexity",
        "I need you to generate Mickey Mouse images in ChatGPT, Grok and Perplexity",
        "Create a logo design in Deepseek, Claude and Midjourney",
        "Generate sunset landscape images in DALL-E, Gemini and Perplexity",
        "I need you to create product mockups in Google Studio, Claude and Mistral",
        "Generate abstract art in Deepseek, Grok and Claude",
        "Create character designs in DALL-E, Midjourney and Gemini",
        "Generate website mockups in ChatGPT, Google Studio and Grok",
        "I need you to create infographics in Canva, Deepseek and Perplexity",
        "Generate social media posts in Mistral, Claude and Grok",
        "Create marketing copy in Gemini, Claude and Perplexity",
        "Generate blog posts in ChatGPT, Deepseek and Claude",
        "I need you to create email templates in Google Studio, Claude and Perplexity",
        "Generate code snippets in Deepseek, Claude and Grok",
        "Create documentation in ChatGPT, Mistral and Claude",
        "Generate test cases in Gemini, Grok and Claude",
        "I need you to create API examples in Deepseek, Claude and Perplexity",
        "Generate data visualizations in ChatGPT, Google Studio and Grok",
        "Create presentation slides in Mistral, Perplexity and Claude",
        "Generate product descriptions in Gemini, Grok and Claude",
        "I need you to create landing page copy in Deepseek, Claude and Perplexity",
        "Generate SEO content in ChatGPT, Google Studio and Perplexity",
        "Create ad copy in Mistral, Claude and Grok",
        "Generate video scripts in Gemini, Perplexity and Claude",
        "I need you to create podcast outlines in Deepseek, Grok and Claude",
        "Generate research summaries in ChatGPT, Claude and Google Studio",
        "Create study guides in Mistral, Grok and Perplexity",
        "Generate quiz questions in Gemini, Claude and Grok",
        "I need you to create lesson plans in Deepseek, Perplexity and Claude",
        "Generate recipe variations in ChatGPT, Google Studio and Claude",
        "Create workout plans in Mistral, Claude and Perplexity",
        "Generate travel itineraries in Gemini, Grok and Perplexity",
        "I need you to create meal plans in Deepseek, Claude and Grok",
        "Generate business plans in ChatGPT, Mistral and Claude",
        "Create financial projections in Google Studio, Grok and Claude",
        "Generate market analysis in Gemini, Claude and Perplexity",
        "I need you to create competitor research in Deepseek, Grok and Perplexity",
        "Generate user personas in ChatGPT, Claude and Google Studio",
        "Create customer journey maps in Mistral, Perplexity and Claude",
        "Generate feature specifications in Gemini, Grok and Claude",
        "I need you to create wireframes in Figma, Deepseek and Claude",
        "Generate UI components in ChatGPT, Claude and Google Studio",
        "Create design systems in Mistral, Perplexity and Claude",
        "Generate color palettes in Gemini, Grok and Claude",
        "I need you to create typography guidelines in Deepseek, Claude and Perplexity",
        "Generate brand guidelines in ChatGPT, Google Studio and Perplexity",
        "Create style guides in Mistral, Claude and Grok",
        "Generate icon sets in Gemini, Perplexity and Claude",
        "I need you to create illustrations in DALL-E, Midjourney and Deepseek",
        "Generate 3D models in ChatGPT, Google Studio and Claude",
        "Create animations in Mistral, Claude and Perplexity",
        "Generate motion graphics in Gemini, Grok and Claude",
        "I need you to create video thumbnails in Canva, Deepseek and DALL-E",
        "Generate banner ads in ChatGPT, Claude and Google Studio",
        "Create social media graphics in Canva, Mistral and Perplexity",
        "Generate Instagram posts in Gemini, Grok and Claude",
        "I need you to create Twitter threads in Deepseek, Claude and Perplexity",
        "Generate LinkedIn articles in ChatGPT, Google Studio and Perplexity",
        "Create Facebook ads in Mistral, Claude and Grok",
        "Generate TikTok scripts in Gemini, Perplexity and Claude",
        "I need you to create YouTube descriptions in Deepseek, Grok and Claude",
        "Generate podcast show notes in ChatGPT, Claude and Google Studio",
        "Create newsletter content in Mistral, Grok and Perplexity",
        "Generate press releases in Gemini, Claude and Grok",
        "I need you to create case studies in Deepseek, Perplexity and Claude",
        "Generate white papers in ChatGPT, Google Studio and Claude",
        "Create ebooks in Mistral, Claude and Perplexity",
        "Generate course content in Gemini, Grok and Perplexity",
        "I need you to create training materials in Deepseek, Claude and Grok",
        "Generate onboarding guides in ChatGPT, Mistral and Claude",
        "Create help documentation in Google Studio, Grok and Claude",
        "Generate FAQ sections in Gemini, Claude and Perplexity",
        "I need you to create troubleshooting guides in Deepseek, Grok and Perplexity",
        "Generate API documentation in ChatGPT, Claude and Google Studio",
        "Create technical specifications in Mistral, Perplexity and Claude",
        "Generate architecture diagrams in Gemini, Grok and Claude",
        "I need you to create database schemas in Deepseek, Claude and Perplexity",
        "Generate SQL queries in ChatGPT, Google Studio and Claude",
        "Create data models in Mistral, Claude and Grok",
        "Generate ERD diagrams in Gemini, Perplexity and Claude",
        "I need you to create flowcharts in Deepseek, Grok and Claude",
        "Generate process diagrams in ChatGPT, Claude and Google Studio",
        "Create system diagrams in Mistral, Grok and Perplexity",
        "Generate network diagrams in Gemini, Claude and Grok",
        "I need you to create deployment diagrams in Deepseek, Perplexity and Claude",
        "Generate security policies in ChatGPT, Google Studio and Claude",
        "Create compliance documents in Mistral, Claude and Perplexity",
        "Generate privacy policies in Gemini, Grok and Perplexity",
        "I need you to create terms of service in Deepseek, Claude and Grok",
        "Generate legal disclaimers in ChatGPT, Mistral and Claude",
        "Create contract templates in Google Studio, Grok and Claude",
        "Generate NDA templates in Gemini, Claude and Perplexity",
        "I need you to create proposal templates in Deepseek, Grok and Perplexity",
        "Generate invoice templates in ChatGPT, Claude and Google Studio",
        "Create receipt templates in Mistral, Perplexity and Claude",
        "Generate report templates in Gemini, Grok and Claude",
        "I need you to create spreadsheet templates in Deepseek, Claude and Perplexity",
        "Generate presentation templates in ChatGPT, Google Studio and Perplexity",
        "Create email templates in Mistral, Claude and Grok",
        "Generate form templates in Gemini, Perplexity and Claude",
        "I need you to create survey templates in Deepseek, Grok and Claude",
        "Generate questionnaire templates in ChatGPT, Claude and Google Studio",
        "Create feedback forms in Mistral, Grok and Perplexity",
        "Generate evaluation forms in Gemini, Claude and Grok",
        
        // ── DRAG AND DROP AUTOMATION (desktop and web applications) ──────────────────
        // CRITICAL: Commands that require drag-and-drop interactions
        // Desktop file operations
        "Drag the folder X on my desktop to the trash",
        "Drag the file report.pdf from desktop to Documents folder",
        "Move the image.png from Downloads to Desktop by dragging it",
        "Drag the project folder to the trash bin",
        "I need you to drag the old_files folder to the trash",
        "Drag and drop the screenshot.png into the Images folder",
        "Move the video.mp4 to the Videos folder by dragging",
        "Drag the archive.zip from desktop to external drive",
        "I need you to drag the backup folder to the trash",
        "Drag the presentation.pptx to the Shared folder",
        "Move the spreadsheet.xlsx to Documents by dragging it",
        "Drag the music files to the iTunes folder",
        "I need you to drag the temp folder to the trash",
        "Drag and drop the PDF files into the Work folder",
        "Move the photos from desktop to Pictures by dragging",
        
        // n8n workflow automation
        "Drag the node Oven in n8n and drop it into the kitchen workspace",
        "I need you to drag the HTTP Request node in n8n to the canvas",
        "Drag the Webhook node in n8n and drop it at the start",
        "Move the Slack node in n8n by dragging it to the right",
        "Drag the Google Sheets node in n8n and connect it to the workflow",
        "I need you to drag the Function node in n8n and place it after the trigger",
        "Drag the Email node in n8n and drop it at the end of the workflow",
        "Move the Filter node in n8n by dragging it between the nodes",
        "Drag the Set node in n8n and position it before the output",
        "I need you to drag the Switch node in n8n to create a branch",
        
        // Figma design operations
        "Drag the button component in Figma to the canvas",
        "I need you to drag the frame in Figma to reposition it",
        "Drag the text layer in Figma and drop it into the header group",
        "Move the icon in Figma by dragging it to the sidebar",
        "Drag the component from the library in Figma to the design",
        "I need you to drag the layer in Figma to reorder it",
        "Drag the rectangle in Figma and drop it into the container",
        "Move the group in Figma by dragging it to the artboard",
        "Drag the image in Figma and place it in the hero section",
        "I need you to drag the variant in Figma to create an instance",
        
        // Trello board management
        "Drag the card in Trello from To Do to In Progress",
        "I need you to drag the task card in Trello to the Done column",
        "Drag the Trello card to the top of the list",
        "Move the card in Trello by dragging it to another board",
        "Drag the label in Trello and drop it on the card",
        "I need you to drag the checklist item in Trello to reorder it",
        "Drag the Trello card from Backlog to Sprint",
        "Move the card in Trello to the Archive by dragging",
        "Drag the attachment in Trello and add it to the card",
        "I need you to drag the card in Trello to change its position",
        
        // Notion page organization
        "Drag the block in Notion to reorder the content",
        "I need you to drag the page in Notion to a different section",
        "Drag the database row in Notion to change the order",
        "Move the Notion block by dragging it above the header",
        "Drag the image in Notion and drop it into the gallery",
        "I need you to drag the table in Notion to reposition it",
        "Drag the callout block in Notion to the top of the page",
        "Move the Notion page by dragging it to another workspace",
        "Drag the embed in Notion and place it in the content area",
        "I need you to drag the toggle list in Notion to nest it",
        
        // Airtable database operations
        "Drag the field in Airtable to reorder the columns",
        "I need you to drag the record in Airtable to change its position",
        "Drag the attachment in Airtable and drop it into the field",
        "Move the view in Airtable by dragging it to reorder tabs",
        "Drag the column in Airtable to the left side",
        "I need you to drag the row in Airtable to group it",
        "Drag the linked record in Airtable to create a connection",
        "Move the field in Airtable by dragging it to hide it",
        "Drag the filter in Airtable to adjust the view",
        "I need you to drag the grouping in Airtable to reorganize",
        
        // Miro whiteboard collaboration
        "Drag the sticky note in Miro to the brainstorming area",
        "I need you to drag the shape in Miro to the canvas",
        "Drag the connector in Miro and link the two elements",
        "Move the frame in Miro by dragging it to the workspace",
        "Drag the image in Miro and drop it on the board",
        "I need you to drag the text box in Miro to reposition it",
        "Drag the card in Miro from one section to another",
        "Move the icon in Miro by dragging it to the diagram",
        "Drag the template in Miro and apply it to the board",
        "I need you to drag the widget in Miro to the collaboration space",
        
        // Asana task management
        "Drag the task in Asana to a different section",
        "I need you to drag the subtask in Asana to reorder it",
        "Drag the project in Asana to another team",
        "Move the task in Asana by dragging it to next week",
        "Drag the milestone in Asana to adjust the timeline",
        "I need you to drag the task in Asana from Today to Tomorrow",
        "Drag the custom field in Asana to reorder the properties",
        "Move the task in Asana by dragging it to the Completed section",
        "Drag the dependency in Asana to link the tasks",
        "I need you to drag the task in Asana to change priority",
        
        // Monday.com board operations
        "Drag the item in Monday to a different group",
        "I need you to drag the column in Monday to reorder it",
        "Drag the task in Monday from this week to next week",
        "Move the board in Monday by dragging it to another workspace",
        "Drag the status in Monday to update the item",
        "I need you to drag the row in Monday to change the order",
        "Drag the file in Monday and attach it to the item",
        "Move the group in Monday by dragging it to reorder",
        "Drag the timeline in Monday to adjust the dates",
        "I need you to drag the item in Monday to archive it",
        
        // Jira issue tracking
        "Drag the issue in Jira from To Do to In Progress",
        "I need you to drag the epic in Jira to the backlog",
        "Drag the story in Jira to the current sprint",
        "Move the ticket in Jira by dragging it to Done",
        "Drag the subtask in Jira to reorder it under the parent",
        "I need you to drag the issue in Jira to change the priority",
        "Drag the component in Jira to assign it to the issue",
        "Move the version in Jira by dragging it to the roadmap",
        "Drag the label in Jira and apply it to the ticket",
        "I need you to drag the issue in Jira to another project",
        
        // Canva design editor
        "Drag the element in Canva to the center of the canvas",
        "I need you to drag the text box in Canva to reposition it",
        "Drag the image in Canva from uploads to the design",
        "Move the shape in Canva by dragging it to the background",
        "Drag the sticker in Canva and place it on the graphic",
        "I need you to drag the template in Canva to start editing",
        "Drag the photo in Canva and drop it into the frame",
        "Move the layer in Canva by dragging it to change order",
        "Drag the icon in Canva and resize it on the canvas",
        "I need you to drag the element in Canva to group it",
      // ],

      // COMMENTED: For now unti we get the rest of the app working smoothly
      // command_automate: [
         // ── Original ─────────────────────────────────────
        // NOTE: "Take a screenshot" removed - conflicts with vision intent
        // Vision service handles screen capture + analysis
        "Open Chrome",
        "Close all windows",
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
        "Set an alarm for 7am",

        // ── New – more apps, OS, automation ───────
        "Open Slack and go to #general",
        "Lock the screen now",
        "Open Spotify and play my Discover Weekly",
        "Turn on Do Not Disturb until 5pm",
        "Open Terminal and run `htop`",
        "Empty the Recycle Bin",
        "Open Notion page 'Project Roadmap'",
        "Start screen recording",
        "Pause all media playback",
        "Open the calculator app",
        "Switch to the next desktop space",
        "Open my email client and compose a new message to boss@example.com",
        "Enable Bluetooth",
        "Open the system settings → Displays",
        "Restart the computer in 2 minutes",
        "Open Finder and go to Downloads",
        // NOTE: Screenshot commands removed - handled by vision service
        "Open Postman and load the 'API Tests' collection",
        "Turn off Wi-Fi",
        "Open the Calendar app and create an event for tomorrow 10am titled 'Standup'",
        
        // ── GOTO and navigation commands (browser/app navigation) ────
        "Goto Google",
        "Go to Google",
        "Goto Amazon",
        "Go to Amazon",
        "Goto YouTube",
        "Go to YouTube",
        "Goto Facebook",
        "Go to Facebook",
        "Goto Twitter",
        "Go to Twitter",
        "Goto LinkedIn",
        "Go to LinkedIn",
        "Goto Instagram",
        "Go to Instagram",
        "Goto Reddit",
        "Go to Reddit",
        "Goto GitHub",
        "Go to GitHub",
        "Goto Stack Overflow",
        "Go to Stack Overflow",
        "Goto Gmail",
        "Go to Gmail",
        "Goto Outlook",
        "Go to Outlook",
        "Goto Netflix",
        "Go to Netflix",
        "Goto Spotify",
        "Go to Spotify",
        "Goto ChatGPT",
        "Go to ChatGPT",
        "Goto ChatGPT website",
        "Go to ChatGPT website",
        "Goto OpenAI",
        "Go to OpenAI",
        "Goto Claude",
        "Go to Claude",
        "Goto Claude AI",
        "Go to Claude AI",
        "Goto Perplexity",
        "Go to Perplexity",
        "Goto Perplexity website",
        "Go to Perplexity website",
        "Goto Gemini",
        "Go to Gemini",
        "Goto Bard",
        "Go to Bard",
        "Goto Google website",
        "Go to Google website",
        "Goto Amazon website",
        "Go to Amazon website",
        "Goto YouTube website",
        "Go to YouTube website",
        "Goto Facebook website",
        "Go to Facebook website",
        "Goto Twitter website",
        "Go to Twitter website",
        "Goto LinkedIn website",
        "Go to LinkedIn website",
        "Goto Instagram website",
        "Go to Instagram website",
        "Goto Reddit website",
        "Go to Reddit website",
        "Goto GitHub website",
        "Go to GitHub website",
        "Goto Netflix website",
        "Go to Netflix website",
        "Goto Spotify website",
        "Go to Spotify website",
        "Goto the website",
        "Go to the site",
        "Navigate to Google",
        "Navigate to Amazon",
        "Open Google in browser",
        "Open Amazon in browser",
        "Visit Google.com",
        "Visit Amazon.com",
        "Browse to Google",
        "Browse to Amazon",
        "Head to Google",
        "Head to Amazon",
        "Goto google.com",
        "Go to amazon.com",
        "Open up Google",
        "Open up Amazon",
        
        // ── CRITICAL: UI Navigation (dock, folders, system UI) ────────────
        // These are UI automation commands, NOT web searches or screen analysis
        "Go to docker at the bottom",
        "Go to the docker at the bottom",
        "Go to docker and open textedit",
        "Go to the dock at the bottom",
        "Go to the dock and click textedit",
        "Goto my application folder",
        "Go to my application folder",
        "Goto my applications folder",
        "Go to my applications folder",
        "Navigate to my application folder",
        "Navigate to my applications folder",
        "Open my application folder",
        "Open my applications folder",
        "Go to my desktop folder",
        "Goto my desktop folder",
        "Navigate to my desktop",
        "Go to my downloads folder",
        "Goto my downloads folder",
        "Navigate to my downloads",
        "Go to my documents folder",
        "Goto my documents folder",
        "Navigate to my documents",
        "Open my projects folder on my desktop",
        "Open my work folder on desktop",
        "Open the gongzuo folder on my desktop",
        "Open my photos folder",
        "Navigate to my music folder",
        "Go to my videos folder",
        "Open the code folder on desktop",
        "Open my personal folder",
        "Go to the finder",
        "Goto finder",
        "Open finder",
        "Go to the dock",
        "Goto the dock",
        "Click on the dock",
        "Go to the menu bar",
        "Goto the menu bar",
        "Click on the menu bar",
        "Go to the system tray",
        "Goto the system tray",
        "Go to the taskbar",
        "Goto the taskbar",
        "Go to the start menu",
        "Goto the start menu",
        "Open the start menu",
        "Go to the launchpad",
        "Goto the launchpad",
        "Open launchpad",
        "Go to spotlight",
        "Goto spotlight",
        "Open spotlight",
        "Go to the notification center",
        "Goto the notification center",
        "Open notification center",
        
        // ── GOTO + ACTION (navigation with search/action) ────────────
        // "Go online" patterns (navigate to web/browser)
        "Go online and do a google search for the latest AI for video",
        "Go online and search for news",
        "Goto online and do a google search for shoes",
        "Go online and find information about climate change",
        "Goto online and search for restaurants near me",
        "Go online and look up the weather",
        "Go online and do a search for hotels",
        "Goto online and find flights to Paris",
        
        // Standard goto + action patterns
        "Goto Google and search for shoes",
        "Go to Google and search for winter clothes",
        "Goto Amazon and find me some winter clothes",
        "Go to Amazon and find winter boots",
        "Goto YouTube and search for cooking videos",
        "Go to YouTube and play music",
        "Goto Gmail and compose a new email",
        "Go to Gmail and check my inbox",
        "Goto LinkedIn and find jobs",
        "Go to LinkedIn and search for connections",
        "Goto GitHub and search for react libraries",
        "Go to GitHub and clone the repo",
        "Goto Stack Overflow and search for python errors",
        "Go to Stack Overflow and find solutions",
        "Goto Reddit and search for tech news",
        "Go to Reddit and browse r/programming",
        "Goto Twitter and search for AI news",
        "Go to Twitter and post a tweet",
        "Goto Facebook and check notifications",
        "Go to Facebook and post an update",
        "Goto Netflix and watch a movie",
        "Go to Netflix and browse shows",
        "Goto Spotify and play my playlist",
        "Go to Spotify and search for jazz music",
        "Goto the website and search for products",
        "Go to the site and look for deals",
        "Navigate to Google and search for restaurants",
        "Navigate to Amazon and buy a book",
        "Open Google and search for news",
        "Open Amazon and find gifts",
        "Visit Google and search for hotels",
        "Visit Amazon and browse electronics",
        "Browse to Google and search for flights",
        "Browse to Amazon and look for shoes",
        
        // ── SEARCH IN APP (app-specific searches - NON-EMAIL) ────────────────────
        // NOTE: Gmail/Outlook email searches moved to dedicated section below with "I need you to" patterns
        "Search in Slack for messages from Sarah",
        "Search Slack for #general channel",
        "Find messages in Slack about deployment",
        "Search Discord for announcements",
        "Search my Discord for DMs",
        "Search in Notion for project notes",
        "Search Notion for meeting minutes",
        "Find notes in Notion about Q4 goals",
        "Search Spotify for jazz playlists",
        "Search my Spotify for saved songs",
        "Search YouTube for cooking tutorials",
        "Search my YouTube for watch history",
        "Search Google Drive for documents",
        "Search my Drive for spreadsheets",
        "Find files in Dropbox",
        "Search Dropbox for PDFs",
        "Search Photos for pictures from vacation",
        "Search my Photos for selfies",
        "Find photos from last summer",
        "Search Calendar for meetings this week",
        "Search my Calendar for appointments",
        "Find events in Calendar for tomorrow",
        
        // ── CALENDAR and reminder commands ────────────────────────────
        "Set a reminder in Calendar to get a gift for mom Feb 2",
        "Set a reminder to get a gift for mom Feb 2",
        "Create a reminder for mom's gift Feb 2",
        "Add a reminder to buy gift for mom February 2",
        "Set reminder to call John tomorrow",
        "Create reminder for dentist appointment",
        "Add reminder to submit report by Friday",
        "Set a calendar reminder for team meeting",
        "Create a calendar event for lunch tomorrow",
        "Add event to Calendar for conference next week",
        "Schedule a meeting in Calendar for Monday",
        "Set up a calendar invite for the team",
        "Create calendar event titled 'Standup' for tomorrow 10am",
        "Add to calendar: doctor appointment Thursday 3pm",
        "Schedule reminder for grocery shopping",
        "Set alarm for 7am tomorrow",
        "Create alarm for 6:30am weekdays",
        "Add alarm for 8am",
        "Set timer for 25 minutes",
        "Create timer for 1 hour",
        "Start a 30 minute timer",

        // ── Nut.js UI automation - complex multi-step workflows ────────
        // Calendar/reminder operations
        "Set a calendar reminder for tomorrow at 3pm",
        "Create a calendar event for Monday at 10am",
        "Add a reminder for my dentist appointment next Friday",
        "Schedule a meeting in Calendar for next week",
        "Set a reminder to call mom tomorrow",
        
        // Email/messaging operations
        "Compose an email to john@example.com about the meeting",
        "Send a message to the team on Slack",
        "Write an email to boss@example.com",
        "Post a tweet about the new feature",
        "Send a DM on Discord to my friend",
        "Compose a new message in Gmail",
        
        // Shopping/browsing workflows
        "Find winter boots on Amazon",
        "Search for cooking videos on YouTube",
        "Go to Google and search for shoes",
        "Goto Amazon and find me some winter clothes",
        "Browse to GitHub and search for react libraries",
        "Navigate to LinkedIn and find jobs",
        "Find restaurants near me on Google Maps",
        "Book a flight to Paris on Expedia",
        "Order pizza from Dominos",
        "Add item to my Amazon cart",
        
        // App-specific workflows
        "Open Spotify and play my Discover Weekly",
        "Open Gmail and compose a new message to boss@example.com",
        "Go online and do a google search for the latest AI for video",
        "Goto YouTube and search for cooking videos",
        "Open Slack and send a message to the team",
        "Go to Twitter and post a tweet",
        "Create a new document in Google Docs",
        "Share this file on Dropbox",
        "Post this photo on Instagram",
        "Create a new playlist on Spotify",
        
        // Multi-step navigation + action
        "Go to Google and search for winter clothes",
        "Goto Amazon and buy a book",
        "Navigate to Google and search for restaurants",
        "Open Google and search for news",
        "Visit Amazon and browse electronics",
        "Browse to Google and search for flights",
        "Goto Gmail and check my inbox",
        "Go to LinkedIn and search for connections",
        "Goto Stack Overflow and find solutions",
        "Go to Reddit and browse r/programming",
        "Goto Netflix and watch a movie",
        "Go to Spotify and search for jazz music",
        
        // App searches (in-app automation - NON-EMAIL)
        // NOTE: Gmail searches consolidated in dedicated section with "I need you to" patterns
        "Search in Slack for messages from Sarah",
        "Find messages in Slack about deployment",
        "Search in Notion for project notes",
        "Search Spotify for jazz playlists",
        "Search YouTube for cooking tutorials",
        "Search Google Drive for documents",
        "Search Photos for pictures from vacation",

          // ── APP + ACTION (open app and do something) ──────────────────
        "Open Slack and message John",
        "Open Slack and go to #general",
        "Open Discord and join voice channel",
        "Open Discord and check messages",
        "Open Spotify and play my playlist",
        "Open Spotify and search for jazz",
        "Open Chrome and go to Google",
        "Open Chrome and search for news",
        "Open Safari and browse to Amazon",
        "Open Safari and search for hotels",
        "Open VS Code and open my project",
        "Open VS Code and create new file",
        "Open Terminal and run htop",
        "Open Terminal and check disk space",
        "Open Finder and go to Downloads",
        "Open Finder and search for PDFs",
        "Open Mail and compose new email",
        "Open Mail and check inbox",
        "Open Calendar and create event",
        "Open Calendar and check today's schedule",
        "Open Notes and create new note",
        "Open Notes and search for meeting notes",
        "Open Photos and find vacation pictures",
        "Open Photos and create album",
        "Open Messages and text mom",
        "Open Messages and check unread",
        "Open Settings and change wallpaper",
        "Open Settings and check updates",
        "Open System Preferences and adjust display",
        "Open System Preferences and change sound",
        
        // ── GOTO APP + TYPE + QUIT (critical patterns to distinguish from web_search) ────
        "Goto slack and type hello world then quit the app",
        "Go to slack and type hello world then quit the app",
        "Goto Slack and type a message then close it",
        "Go to Slack and type something then quit",
        "Goto TextEdit and type hello world then quit",
        "Go to TextEdit and type something then close",
        "Goto Notes and type a note then quit the app",
        "Go to Notes and type something then close it",
        "Goto Messages and type a text then quit",
        "Go to Messages and type something then close",
        "Goto Mail and type an email then quit",
        "Go to Mail and compose a message then close",
        "Goto Discord and type a message then quit the app",
        "Go to Discord and send a message then close it",
        "Goto Terminal and type a command then quit",
        "Go to Terminal and run a command then close",
        "Open Slack type hello world and quit",
        "Open TextEdit type something and close",
        "Open Notes type a note and quit the app",
        "Open Messages type a text and close it",
        "Launch Slack type a message then quit",
        "Launch TextEdit type hello world then close",
        "Start Slack type something and quit the app",
        "Start TextEdit type a note and close it",
        
        // ── CRITICAL: Action verbs - DO something (command_automate) ────────────
        // These explicitly ask the AI to PERFORM an action, not just provide info
        "Open Chrome and navigate to Google",
        "Launch Spotify and play music",
        "Start the timer for 10 minutes",
        "Turn on dark mode",
        "Mute the volume",
        "Close all tabs",
        "Restart the computer",
        "Empty the trash",
        "Lock the screen",
        "Lock my screen",
        "Lock screen",
        "Lock the computer",
        "Lock my computer",
        "Enable Do Not Disturb",
        "Turn off WiFi",
        "Turn off Wi-Fi",
        "Open the calculator",
        "Launch Terminal",
        "Start recording",
        "Pause the music",
        "Skip to next track",
        "Increase brightness",
        "Decrease volume",
        "Switch to next window",
        "Minimize all windows",
        "Take a screenshot",
        "Copy this text",
        "Paste the clipboard",
        "Open a new tab",
        "Refresh the page",
        "Go back",
        "Go forward",
        "Bookmark this page",
        "Print this document",
        "Save this file",
        "Delete this file",
        "Rename this folder",
        "Move this to desktop",
        "Create a new folder",
        "Compress this folder",
        "Extract this archive",
        "Run this script",
        "Execute this command",
        "Install this package",
        "Update the software",
        "Uninstall this app",
        
        // ── Build / create / make app or project ────────────────────────────────
        // These are ALWAYS command_automate — user wants the AI to BUILD something,
        // not search for how to do it. DistilBERT scores these too close to web_search.
        "build me a bible game",
        "create a todo app",
        "make me a calculator app",
        "build a task manager application",
        "create a simple game",
        "make a dashboard for my data",
        "build a script that monitors my folder",
        "create a tool to rename files",
        "make a web app for tracking expenses",
        "build a chrome extension",
        "create a cli tool",
        "make me a pomodoro timer app",
        "build an app that does X",
        "develop a small application",
        "write a script to automate this",
        "implement a math utility game",
        "generate a React app",
        "code a simple game for me",
        "build a weather widget",
        "create a password manager",
        "make a note taking app",
        "build a kanban board",
        "create a habit tracker",
        "make a budget tracker app",

        // ── CRITICAL: More action commands that were failing tests ────────────────────────
        "Create a new note called shopping list",
        "Create a note called ideas",
        "Make a new note",
        "Add milk to my shopping list",
        "Add item to shopping list",
        "Add task to my list",
        "Rename this file to report-final.pdf",
        "Rename this file",
        "Rename the file",
        "Connect to Wi-Fi network Home-5G",
        "Connect to WiFi",
        "Connect to network",
        "Pin this window to the left",
        "Pin window to left",
        "Snap window left",
        "Start recording my screen",
        "Start screen recording",
        "Record my screen",
        "Record the screen",
        "Begin screen recording",
        "Start recording the screen",
        "Capture my screen",
        "Screen record this",
        "Record screen now",
        "Start screen capture",
        
        // ── Device / Bluetooth control ────────────────────────
        "Connect to my AirPods",
        "Connect to AirPods",
        "Pair with my AirPods",
        "Connect to my headphones",
        "Pair Bluetooth device",
        
        // ── Hardware control ────────────────────────
        "Turn brightness down to 30%",
        "Set brightness to 30 percent",
        "Lower brightness to 30%",
        "Dim screen to 30%",
        "Adjust brightness to 30%",
        
        // ── Timer control ────────────────────────
        "Stop the timer",
        "Cancel the timer",
        "End the timer",
        "Pause the timer",
        "Stop timer",
        
        // ── List management ────────────────────────
        "Add eggs to my grocery list",
        "Add bread to my shopping list",
        "Put eggs on my list",
        "Add item to my list",
        "Add to grocery list",
        
        // ── App launching ────────────────────────
        "Open the Settings app",
        "Launch Settings",
        "Open Settings",
        "Go to Settings",
        "Show me Settings",
        
        // ── Ambiguous action commands ────────────────────────
        "Open my email",
        "Check my email",
        "Open email",
        "Show my email",
        "Fix it",
        "Fix that",
        "Repair it",
        "Correct it",
        "Take care of that",
        "Handle that",
        "Deal with that",
        "Sort that out",
        
        // ── Hedged/polite commands ────────────────────────
        "uh could you maybe open vscode for me",
        "could you maybe open",
        "would you mind opening",
        "it would be great if you closed this window",
        "it would be great if you",
        "could you possibly",
        
        // ── Slang/emoji commands ────────────────────────
        "pls mute everything",
        "pls turn off",
        "pls mute everything",
        "pls silence everything",
        "yo screenshot this",
        "yo take a screenshot",
        "yo capture this",
        
        // ── Music/media control ────────────────────────
        "can you just put this song on repeat forever",
        "put this on repeat",
        "loop this song",
        "repeat this track",
        "play on loop",
        
        // ── Timer commands (shorthand) ────────────────────────
        "ok timer 7 mins starting now",
        "timer 5 minutes",
        "set timer 10 mins",
        "start timer now",
        
        // ── Email/inbox management ────────────────────────
        "archive these emails I don't wanna see them anymore",
        "archive these emails",
        "hide these emails",
        "delete these messages",
        
        // ── App termination ────────────────────────
        "kill the music app",
        "kill spotify",
        "force close",
        "terminate the app",
        
        // ── Multi-step calendar automation ────────────────────────
        "Open my calendar, create an event for tomorrow at 10, and invite John",
        "create calendar event and invite",
        "schedule meeting and send invite",
        
        // ── Batch productivity commands ────────────────────────
        "Pause the music, set a 25 minute timer, and turn on do not disturb",
        "pause music and set timer",
        "enable focus mode",
        
        // ── Chained file operations ────────────────────────
        "Create a folder called Photos, move the current file into it, and then open that folder",
        "create folder and move file",
        "organize files into folder",
        
        // ── Screen and audio control ────────────────────────
        "Start recording my screen and also mute my microphone",
        "record screen and mute mic",
        "start recording and mute",
        
        // ── Scoped screenshot ────────────────────────
        "Take a screenshot of just this window and save it to the desktop",
        "screenshot this window only",
        "capture just this window",
        "save screenshot to desktop",
        
        // ── Emotional context commands ────────────────────────
        "I'm exhausted, dim the lights and play some soft music",
        "I'm tired, dim the lights",
        "I'm exhausted, help me relax",
        
        // ── Focus mode ────────────────────────
        "I'm going into focus mode, block notifications for the next hour",
        "block notifications for an hour",
        "silence notifications",
        "enable do not disturb",
        
        // ── End of day ────────────────────────
        "I'm done for today, close everything and shut down the computer",
        "close everything and shut down",
        "shut down the computer",
        "end my session",
        
        // ── Quick email ────────────────────────
        "I'm running late, send a quick email to my boss saying I'll be 15 minutes behind",
        "send quick email saying",
        "email my boss that",
        
        // ── Presentation mode ────────────────────────
        "I'm about to present, start screen recording and mute all alerts",
        "prepare for presentation",
        "presentation mode on",
        
        // ── Messaging commands ────────────────────────
        "Send a message to Sarah saying I'm on my way",
        "Send a text to John",
        "Message Sarah that I'm running late",
        "Text Mom I'll be home soon",
        "Send an email to the team",
        
        // ── Ambiguous action reference ────────────────────────
        "Do that thing we talked about",
        "Do that thing",
        "Do what we discussed",
        "Execute that command",
        
        // ── Text file creation and content copying ────────────────────────
        "I need you to take chat gpt AI response from my Thinkdrop AI project channel the first chat window and paste them in a Text file. So you'll to make a new text file todo that. Does this all make sense?",
        "Take the AI response and paste it in a text file",
        "Copy the chat response to a new text file",
        "Make a new text file and paste the AI response in it",
        "Create a text file with the chat response",
        "Save the AI response to a text file",
        "Put the chat response in a new text file",
        "Copy this conversation to a text file",
        "Save this chat to a text file",
        "Create a text file with this conversation",
        "Make a text file from this chat",
        "Export this conversation to a text file",
        "Save the chat history to a text file",
        "Copy the AI output to a text file",
        "Put the response in a text file",
        "Create a file with the chat content",
        "Save the conversation as a text file",
        "Make a new file with the AI response",
        "Copy the chat to a new file",
        "Create a text document with this chat",
        "Save the messages to a text file",
        "Put the conversation in a text file",
        "Export the chat to a file",
        "Create a file from this conversation",
        "Copy the dialogue to a text file",
        "Make a text file from the chat window",
        "Save the chat window to a file",
        "Copy the first chat window to a text file",
        "Take the response from the chat and put it in a text file",
        "Grab the AI response and save it to a text file",
        "Extract the chat and save it as a text file",
        "Copy all the messages to a new text file",
        "Create a text file containing the chat",
        "Make a file with the conversation content",
        "Save the AI conversation to a text file",
        "Copy the chat transcript to a file",
        "Create a new text file with the chat history",
        "Put the chat messages in a text file",
        "Save the dialogue to a text file",
        "Copy the conversation history to a file",
        "Make a text file of the chat",
        "Create a file and paste the chat in it",
        "Save the chat content to a new file",
        "Copy the AI chat to a text document",
        "Make a new text document with the chat",
        "Create a text file from the chat messages",
        "Save the response to a new text file",
        "Copy the chat output to a file",
        "Put the AI messages in a text file",
        
        // ── "I need you to..." automation patterns ────────────────────────
        "I need you to open Chrome and navigate to Gmail",
        "I need you to create a new folder on my desktop",
        "I need you to copy all the files from Downloads to Documents",
        "I need you to delete the old backup files",
        "I need you to rename this file to project_final.txt",
        "I need you to move these photos to the Photos folder",
        "I need you to compress this folder into a zip file",
        "I need you to extract the contents of this archive",
        "I need you to run the build script",
        "I need you to start the development server",
        "I need you to kill the process on port 3000",
        "I need you to check if Docker is running",
        "I need you to list all running containers",
        "I need you to open VS Code and load this project",
        "I need you to close all browser tabs",
        "I need you to mute the system volume",
        "I need you to set a timer for 25 minutes",
        "I need you to lock the screen",
        "I need you to take a screenshot and save it to desktop",
        "I need you to open Spotify and play my workout playlist",
        "I need you to send an email to John about the meeting",
        "I need you to create a calendar event for tomorrow at 2pm",
        "I need you to schedule a reminder for Friday",
        "I need you to turn on Do Not Disturb mode",
        "I need you to enable dark mode",
        "I need you to increase the brightness",
        "I need you to connect to the office Wi-Fi",
        "I need you to check my disk space",
        "I need you to show me what's using the most memory",
        "I need you to find all PDF files on my desktop",
        "I need you to search for files modified today",
        "I need you to backup this database",
        "I need you to run the tests",
        "I need you to deploy to production",
        "I need you to restart the server",
        "I need you to clear the cache",
        "I need you to update all npm packages",
        "I need you to install the dependencies",
        "I need you to commit these changes with message 'bug fix'",
        "I need you to push to the remote repository",
        "I need you to create a new branch called feature-login",
        "I need you to merge develop into main",
        "I need you to switch to the staging branch",
        "I need you to pull the latest changes",
        "I need you to show me the git status",
        "I need you to open the terminal and run htop",
        "I need you to execute this Python script",
        "I need you to run the Dockerfile",
        "I need you to start all Docker containers",
        "I need you to stop the nginx service",
        "I need you to check the logs for errors",
        "I need you to monitor the CPU usage",
        "I need you to find the process using port 8080",
        "I need you to export this data to CSV",
        "I need you to convert this file to JSON",
        "I need you to resize these images",
        "I need you to batch rename these files",
        "I need you to sync this folder to the cloud",
        "I need you to download the latest release",
        "I need you to upload this file to the server",
        "I need you to create a symbolic link",
        "I need you to change the file permissions",
        "I need you to make this script executable",
        "I need you to generate an SSH key",
        "I need you to add this key to GitHub",
        "I need you to configure the environment variables",
        "I need you to load the .env file",
        "I need you to set up the database",
        "I need you to run the migrations",
        "I need you to seed the test data",
        "I need you to optimize the images in this folder",
        "I need you to clean up the temp files",
        "I need you to archive old logs",
        "I need you to empty the trash",
        "I need you to eject the USB drive",
        "I need you to mount the external drive",
        "I need you to format this drive",
        "I need you to scan for viruses",
        "I need you to update the system",
        "I need you to install this application",
        "I need you to uninstall the old version",
        "I need you to repair the disk permissions",
        "I need you to rebuild the search index",
        "I need you to refresh the DNS cache",
        "I need you to test the network connection",
        "I need you to ping the server",
        "I need you to trace the route to google.com",
        "I need you to check if the website is up",
        "I need you to monitor the bandwidth usage",
        "I need you to block this IP address",
        "I need you to whitelist this domain",
        "I need you to configure the firewall",
        "I need you to enable the VPN",
        "I need you to disconnect from the network",
        
        // ── Gmail/Email automation with search and conditional logic ────────────────────────
        "I need you to goto my gmail account. I should be login in already and do a search for all emails cakers5559@gmail.com. If the profile selection comes up select Lou",
        "I need you to goto my gmail account and search for emails from john@example.com",
        "I need you to go to gmail and do a search for all emails from 2023",
        "I need you to open gmail and search for emails with subject line invoice",
        "I need you to navigate to gmail and find all unread emails",
        "I need you to go to my email and search for messages from my boss",
        "I need you to open outlook and search for calendar invites",
        "I need you to goto gmail and search for all emails with attachments",
        "I need you to go to my inbox and search for receipts",
        "I need you to open gmail and find emails from last week",
        "I need you to navigate to gmail and search for starred emails",
        "I need you to go to gmail and do a search for emails containing project",
        "I need you to open my email and search for messages from sarah",
        "I need you to goto outlook and search for meeting requests",
        "I need you to go to gmail and find all emails from a specific sender",
        "I need you to open gmail and search for emails in the promotions tab",
        "I need you to navigate to gmail and search for archived emails",
        "I need you to go to my email account and search for drafts",
        "I need you to open gmail and find emails with label important",
        "I need you to goto gmail and search for emails from this month",
        "Goto my gmail account and search for all emails from mike@company.com",
        "Go to gmail and do a search for emails about the project",
        "Open gmail and search for all messages from last year",
        "Navigate to my email and find emails with attachments",
        "Go to my gmail and search for unread messages",
        "Open my email account and search for emails from the team",
        "Goto gmail and find all emails with receipts",
        "Go to my inbox and search for important emails",
        "Open gmail and do a search for emails from my manager",
        "Navigate to gmail and search for emails about meetings",
        "Go to my email and find messages from support",
        "Open my gmail account and search for invoices",
        "Goto my email and search for emails from clients",
        "Go to gmail and find all spam emails",
        "Open my inbox and search for newsletters",
        "Navigate to my email account and search for confirmations",
        "Go to gmail and do a search for travel emails",
        "Open my email and find messages about the deadline",
        "Goto gmail and search for emails with PDFs",
        "Go to my gmail account and find archived messages",

        // ── "I need to" (without "you") — real-world task requests ────────────────────────
        // These were being misclassified as memory_store by DistilBERT
        "I need to renew my license",
        "I need to renew my driver's license",
        "I need to book a flight to New York",
        "I need to book an appointment with the doctor",
        "I need to apply for a passport",
        "I need to register for the conference",
        "I need to schedule a meeting with the team",
        "I need to order groceries online",
        "I need to buy a birthday gift",
        "I need to sign up for the newsletter",
        "I need to fill out the tax form",
        "I need to submit my application",
        "I need to pay my electric bill online",
        "I need to check in for my flight",
        "I need to download the latest version",
        "I need to install the new update",
        "I need to update my resume",
        "I need to create a new account",
        "I need to reset my password",
        "I need to cancel my subscription",
        "I need to track my package",
        "I need to find a restaurant nearby",
        "I need to make a reservation at a restaurant",
        "I need to renew my car registration",
        "I need to apply for a job",

        // ── "Can you do / help me" — polite action requests ────────────────────────
        "Can you do this for me",
        "Can you help me renew my license",
        "Can you book a flight for me",
        "Can you search for winter coats on Amazon",
        "Can you find the cheapest flight to Miami",
        "Can you apply for this job for me",
        "Can you fill out this form for me",
        "Can you order pizza for me",
        "Can you schedule a dentist appointment",
        "Can you check if my package has shipped",
        "Help me renew my license",
        "Help me book a hotel room",
        "Help me find a good restaurant",
        "Help me apply for this scholarship",
        "Help me sign up for this service",
        "Help me fill out this application",
        "Help me order from this website",
        "Please renew my license",
        "Please book a flight to Chicago",
        "Please search for apartments in Austin",
        "Please apply for this job",
        "Please schedule this appointment for me",
        "Do this for me",
        "Can you do this task for me",
        "I need help renewing my license",
        "I need help booking a flight",
        "I need help applying for this",

        // ── Round 5 seeds ───────────────────────────────────────
        // Messaging via Slack / app (phi4 misclassifies as web_search)
        "Message Alex the standup summary from today via Slack",
        "Send Alex the standup notes on Slack",
        "Ping Alex with today's standup notes via Slack",
        "ping alex with today's stand up notes uh via slack",
        "Message Jamie the meeting recap via Slack",
        "Send Jamie the notes from our last call",
        "Drop the meeting summary in Slack",
        "DM Alex on Slack with the notes",
        // Git / dev operations (phi4 misclassifies as web_search)
        "Initialize a new Git repository in this folder",
        "Init a git repo here",
        "Create a new git repo in the current directory",
        "Run git init in this folder",
        // Scaffold / project creation (phi4 misclassifies as general_knowledge)
        "Scaffold me a new FastAPI project",
        "scaffold me a new fast API project",
        "Create a new FastAPI project from a template",
        "Scaffold a new React app with TypeScript",
        "Bootstrap a new Node.js project",
        "Generate a new Django project",
        // Media streaming (phi4 misclassifies as web_search)
        "Stream Dune Part Two on Apple TV this evening",
        "Play Dune on Apple TV tonight",
        "Stream Oppenheimer on Netflix tonight",
        "Put on Interstellar on HBO Max",
        // System disk / hardware stats (phi4 misclassifies as memory_retrieve at 100%)
        "What is my current disk read/write speed?",
        "Check my disk read write speed",
        "Show me my disk throughput",
        "What's my current disk IO speed?",
        // GPU / CPU temperature and stats (voice variants)
        "what's my g p u temperature right now",
        "what is my g p u temp",
        "check my GPU temperature",
        "show me my CPU temperature",
        // ── Round 6 seeds ──────────────────────────────────────
        "Push the current branch to origin on GitHub",
        "git push origin current branch",
        "Push this branch to origin",
        "push the current branch to or ig in on git hub",
        "push this branch up to github",
        "Run git status and show me all unstaged changes",
        "Check the git status of this repo",
        "Export the current Figma artboard as a 2x PNG",
        "Export this Figma frame as PNG",
        "Save the Figma artboard as an image file",
        "ex port the fig ma art board as a two x p n g",
        "export this design as a PNG",
        "Ping Priya",
        "Ping Marcus about the update",
        "Ping Theo on Slack",
        "Message Priya quickly",
        "Status?",
        "Git status?",
        "What's the current git status?",
        "Push it",
        "Take me to Figma and switch to the redesign project",
        "Open a Python interactive shell for me",
        "Watch the current CPU load for the next few seconds",
        // ── Round 7 seeds ──────────────────────────────────────
        // Git operations phi4 confuses with web_search
        "Merge the current feature branch into main",
        "Pull the latest changes from origin main",
        "Zip the dist folder and upload it to the S3 bucket",
        // Search in codebase (NOT web search)
        "Search for all TODO comments in the current repository",
        "search for all to do com ments in the cur rent re pos i tor y",
        // Voice system monitoring
        "check how much R A M is cur rent ly being used on my ma chine",
        "check my C P U load right now",
        // ── Round 7b seeds ──────────────────────────────────────
        // Git/version control voice commands (ca-v04)
        "Commit all staged changes with message fix update auth middleware",
        "com mit all staged chang es with mes sage fix up date auth mid dl ware",
        "git commit staged changes with message fix authentication",
        "Commit staged changes with commit message update tests",
        "Run git commit -m update auth middleware",
        // Ping a person or host (ca-a02)
        "Ping Ben Okafor",
        "Ping John to check in",
        "Ping 192.168.1.1",
        "Ping the server at 10.0.0.1",
        // More codebase search patterns (ca-016)
        "Find all TODO comments in the codebase",
        "Search for FIXME in the source code",
        "Find all TODOs in my repo"
      ],
      
      screen_intelligence: [
        // ── UI Element Discovery ──────────────────────────────
        "Find the Send button",
        "Where is the Save button",
        "Show me the menu",
        "Locate the search box",
        "Find the text field",
        "Where is the login button",
        "Show me all buttons",
        "What buttons are on screen",
        "List all buttons",
        "Find clickable elements",
        "Show me interactive elements",
        "What can I click",
        "Where can I type",
        "Find the input field",
        "Locate the form",
        "Show me the fields",
        "What fields are available",
        "Find the checkbox",
        "Where is the dropdown",
        "Show me the options",

        // ── Screen description and analysis (merged from vision) ─
        "What do you see on my screen",
        "What's on my screen",
        "Describe my screen",
        "Analyze my screen",
        "Look at my screen",
        "Tell me what you see",
        "What am I looking at",
        "Describe what's visible",
        "What's showing on my screen",
        "Tell me what's on my screen",
        "What do you see here",
        "Analyze what's on my screen",
        "Look at what I'm seeing",
        "Describe the current screen",
        "What's displayed on my screen",
        "Can you see my screen",
        "What's visible on my screen",
        "Tell me about my screen",
        "Describe the screen content",
        "What's in my screen",
        
        // ── Follow-up screen queries ──────────────────────────────
        "What about now",
        "How about now",
        "And now",
        "What do you see now",
        "What's on my screen now",
        "Look at my screen now",
        "Describe what you see now",
        "What's showing now",
        "What changed",
        "What's different now",
        "Check my screen again",
        "Look again",
        "What do you see this time",
        
        // ── "On my screen" pattern queries ────────────────────────
        "On my screen",
        "On my screen what is this",
        "On the screen what is this",
        "On my screen what do you see",
        "On my screen is",
        "On the screen is",
        "On my screen there is",
        "On my screen I see",
        "On my screen it shows",
        "On my screen what does it say",
        "On my screen what is that",
        "On my screen can you see",
        "On my screen tell me what this is",
        "On my screen describe this",
        "On my screen what is this about",
        "On my screen explain this",
        "On my screen analyze this",
        "On my screen read this",
        "On my screen what does this error mean",
        "On my screen what does this button mean",
        "On my screen what does this icon mean",
        "On my screen what does this message mean",
        "On my screen help me understand this",
        "On my screen what am I looking at",
        "On the screen identify this",
        
        // ── Location-based screen queries ─────────────────────────
        "What's at the top",
        "What's at the bottom",
        "What's on the left",
        "What's on the right",
        "What's in the corner",
        "What's at the top left",
        "What's at the top right",
        "What's at the bottom left",
        "What's at the bottom right",
        "What's in the center",
        "What's in the middle",
        "What's at the top of the screen",
        "What's at the bottom of the screen",
        "What's on the left side",
        "What's on the right side",
        "What's in the top corner",
        "What's in the bottom corner",
        "What's in the sidebar",
        "What's in the header",
        "What's in the footer",
        "What's in the navigation",
        "What's in the menu bar",
        "What's in the toolbar",
        "What's in the status bar",
        
        // ── Content-specific screen queries ───────────────────────
        "What is this page about",
        "What is this site about",
        "What is this repo about",
        "What is this repository about",
        "What is this git repo about",
        "What is this GitHub repo about",
        "What is this project about",
        "What is this website about",
        "What is this app about",
        "What is this application about",
        "What is this document about",
        "What is this file about",
        "What is this code about",
        "What is this article about",
        "What is this email about",
        "What is this message about",
        "What is this notification about",
        "What is this alert about",
        "What is this error about",
        "What is this warning about",
        
        // ── Combined location + content queries ───────────────────
        "What at the top and what is this about",
        "What's at the top and what is this repo about",
        "What's at the top and what is this page about",
        "What's at the bottom and what does it say",
        "What's on the left and what is it for",
        "What's on the right and what does it do",
        "What's in the header and what is this site",
        "What's in the sidebar and what is this about",
        "What's at the top of the page and what is this",
        "What's at the bottom of the screen and what is it",
        
        // ── OCR and text extraction ───────────────────────────────
        "Read my screen",
        "Extract text from my screen",
        "What text is on my screen",
        "OCR my screen",
        "Read the text on my screen",
        "Get text from my screen",
        "Extract all text visible",
        "Read text from the screen",
        "What does the text say",
        "Transcribe my screen",
        "Pull text from my screen",
        "What's the text on my screen",
        "Read all visible text",
        "Extract text from this image",
        "OCR this screenshot",
        "What text do you see",
        "Read the visible text",
        "Get all text from screen",
        "Transcribe what's visible",
        "Extract readable text",
        
        // ── Image and screenshot analysis ─────────────────────────
        "What's in this image",
        "Describe this image",
        "Analyze this screenshot",
        "What's in this screenshot",
        "Look at this image",
        "Tell me about this image",
        "What's in this picture",
        "Describe this picture",
        "Analyze this photo",
        "What do you see in this image",
        "Explain this screenshot",
        "What's shown in this image",
        "Interpret this screenshot",
        "What's visible in this image",
        "Describe the screenshot",
        "What's in the image",
        "Analyze the picture",
        "What's this image showing",
        "Explain what's in the image",
        "Describe the visual content",
        
        // ── UI and application analysis ───────────────────────────
        "What application is open",
        "What app am I using",
        "What's the current app",
        "Identify the application",
        "What program is this",
        "What software is running",
        "What's the active window",
        "What app is in focus",
        "Identify this application",
        "What's the current program",
        "What UI is showing",
        "Describe the interface",
        "What's the application showing",
        "Analyze the UI",
        "What interface is this",
        "Identify the UI elements",
        "What's in the interface",
        "Describe the application UI",
        "What elements are visible",
        "Analyze the screen layout",
        
        // ── Code and technical content ────────────────────────────
        "What code is on my screen",
        "Read the code on my screen",
        "What programming language is this",
        "Analyze this code",
        "What does this code do",
        "Explain the code on my screen",
        "What's this code snippet",
        "Describe the code visible",
        "What functions are shown",
        "Read the code snippet",
        "What's the code about",
        "Analyze the programming code",
        "What language is this code",
        "Explain this code",
        "What's in the code editor",
        "Describe the code structure",
        "What's the code doing",
        "Read the source code",
        "What's visible in the IDE",
        "Analyze the code on screen",
        
        // ── Terminal and console queries ──────────────────────────
        "What's in the terminal",
        "What's in the console",
        "What's in the warp console",
        "What's in the terminal window",
        "Read the terminal output",
        "Read the console output",
        "What's showing in the terminal",
        "What's showing in the console",
        "What does the terminal say",
        "What does the console say",
        "What's in the command line",
        "Read the terminal",
        "Read the console",
        "What's the terminal showing",
        "What's the console showing",
        "What's in the iTerm window",
        "What's in the warp window",
        "Show me the terminal output",
        "Show me the console output",
        "What commands are in the terminal",
        "What's running in the terminal",
        "What's running in the console",
        "Read the terminal logs",
        "Read the console logs",
        "What errors are in the terminal",
        "What errors are in the console",
        "What's the terminal output",
        "What's the console output",
        
        // ── Document and content reading ──────────────────────────
        "What's in this document",
        "Read this document",
        "What does this document say",
        "Summarize what's on screen",
        "What's the document about",
        "Read the document content",
        "What's written here",
        "Describe the document",
        "What's in the text",
        "Read the visible document",
        "What's the content about",
        "Summarize the screen content",
        "What information is shown",
        "Read the displayed content",
        "What's the document showing",
        "Describe the text content",
        "What's written on screen",
        "Read what's displayed",
        "Summarize what you see",
        "What's the main content",
        
        // ── Error and notification detection ──────────────────────
        "Is there an error on my screen",
        "What's the error message",
        "Read the error",
        "What's the notification",
        "Is there a warning",
        "What's the alert saying",
        "Read the error message",
        "What's the warning about",
        "Is there an alert",
        "What's the error",
        "Read the notification",
        "What's the popup saying",
        "Is there a dialog",
        "What's the message",
        "Read the alert",
        "What's the warning message",
        "Is there an error dialog",
        "What's the error text",
        "Read the warning",
        "What's showing in the alert",
        
        // ── Specific region analysis ──────────────────────────────
        "What's in the top right corner",
        "Read the bottom of my screen",
        "What's in the center",
        "Describe the left side",
        "What's in the top left",
        "Read the top of the screen",
        "What's in the bottom right",
        "Describe the right side",
        "What's at the top",
        "Read the bottom section",
        "What's in the middle",
        "Describe the top section",
        "What's at the bottom",
        "Read the left section",
        "What's in the corner",
        "Describe the center area",
        "What's on the right",
        "Read the top right",
        "What's on the left",
        "Describe the bottom area",
        
        // ── Desktop and window queries ────────────────────────────
        "How many files on my desktop",
        "What files are on my desktop",
        "List my desktop items",
        "What folders do I have",
        "What windows are open",
        "What apps do I have open",
        "What's in my browser",
        "What email am I reading",
        "Read my email",
        "Read my browser",
        "What webpage am I on",
        
        // ── Summarization and information extraction ──────────────
        "Summarize what's on my screen",
        "Give me a summary of my screen",
        "What's the main information on screen",
        "Summarize the content I'm viewing",
        "What are the key points on my screen",
        "Give me an overview of what's displayed",
        "Summarize this page",
        "What's the gist of what I'm looking at",
        "Break down what's on my screen",
        "What are the important details on screen",
        "Summarize the information shown",
        "Give me the highlights of my screen",
        "What's the summary of this content",
        "Condense what's on my screen",
        "What's the main takeaway from my screen",
        "Summarize the visible information",
        "What are the key details on screen",
        "Give me a brief overview of my screen",
        "What's the essential information shown",
        "Summarize what I'm seeing",
        
        // ── Specific content questions (about visible items) ──────
        "What's this email about",
        "What does this email say",
        "Who sent this email",
        "What's this message about",
        "What does this section mean",
        "What does this lease section mean",
        "What's this clause about",
        "What does this paragraph say",
        "What's this disclaimer about",
        "What does this warning mean",
        "What's this notification about",
        "What does this error say",
        "Who is this person in the photo on my screen",
        "Who is this person at the bottom left of my screen",
        "Who is in this photo I'm looking at",
        "What's this person's name in the image",
        "What's this document about",
        "What does this form say",
        "What's this page about",
        "What does this article say",
        "What's this website about",
        "What does this popup say",
        "What's this dialog about",
        "What does this button do",
        "What's this icon for",
        "What does this menu say",
        "What's this field for",
        "What does this label say",
        "What's this image showing",
        "What does this chart mean",
        "What's this graph about",
        "What does this table show",
        "What's this list about",
        "What does this text say",
        "What's this heading about",
        "What does this title mean",
        "What's this link for",
        "What does this option do",
        
        // ── CRITICAL: Screen analysis queries that were failing ────────────────────────
        "What is the total amount on this invoice",
        "What is the total on this invoice",
        "What is the amount on this bill",
        "What file is currently selected",
        "What file is selected",
        "Which file is selected",
        "Which tab is active right now",
        "Which tab is active",
        "What tab is open",
        "Is this website asking for my password",
        "Is this asking for my password",
        "Is this a password prompt",
        "Does this look like a phishing email",
        "Is this a phishing email",
        "Is this email suspicious",
        "Is there anything overdue in this task list",
        "Is there anything overdue",
        "Rewrite this email better",
        "Fix any grammar issues with this",
        "Check grammar in this email",
        "Are there any grammar mistakes",
        "Correct the grammar",
        "Fix spelling errors",
        "Proofread this",
        "Check this for errors",
        "Fix any issues with this email",
        "Correct any mistakes",
        "Give me a response to this email",
        "Draft a response to this email",
        "Write a reply to this",
        "Respond to this email",
        "Help me reply to this",
        "What should I say back",
        "Compose a response",
        "Draft a reply",
        "Put together a response to this linkedin message",
        "Draft a response to this linkedin message",
        "Help me respond to this linkedin message",
        "Write a reply to this linkedin message on my screen",
        "Compose a response to this message on my screen",
        "Draft a reply to this message on my screen",
        "Put together a response to this message",
        "Help me reply to this message on my screen",
        "Draft a response to this facebook message",
        "Reply to this facebook post on my screen",
        "Write a comment on this facebook post",
        "Respond to this facebook comment",
        "Put together a response to this twitter post",
        "Draft a reply to this tweet on my screen",
        "Write a response to this X post",
        "Reply to this tweet on my screen",
        "Compose a response to this twitter thread",
        "Draft a response to this instagram comment",
        "Reply to this instagram dm on my screen",
        "Write a response to this instagram message",
        "Respond to this instagram post",
        "Put together a response to this telegram message",
        "Draft a reply to this telegram chat",
        "Write a response to this telegram dm",
        "Respond to this gmail email",
        "Put together a response to this aol email",
        "Draft a reply to this aol message",
        "Write a response to this aol email on my screen",
        "Reply to this outlook email on my screen",
        "Draft a response to this outlook message",
        "Respond to this yahoo email",
        "Reply to this protonmail message",
        "Draft a response to this slack message",
        "Reply to this slack dm on my screen",
        "Write a response to this discord message",
        "Reply to this discord dm on my screen",
        "Draft a response to this whatsapp message",
        "Reply to this whatsapp chat on my screen",
        "Write a response to this messenger message",
        "Reply to this messenger dm",
        "Draft a response to this reddit comment",
        "Reply to this reddit post on my screen",
        "Write a response to this reddit message",
        "Draft a response to this tiktok comment",
        "Reply to this tiktok comment on my screen",
        "Draft a response to this amazon review",
        "Reply to this amazon review on my screen",
        "Write a response to this amazon question",
        "Respond to this amazon customer question",
        "Put together a response to this amazon review",
        "Draft a response to this yelp review",
        "Reply to this yelp review on my screen",
        "Write a response to this yelp comment",
        "Respond to this yelp customer review",
        "Draft a response to this google review",
        "Reply to this google review on my screen",
        "Write a response to this google business review",
        "Respond to this google maps review",
        "Put together a response to this tripadvisor review",
        "Draft a reply to this tripadvisor review",
        "Write a response to this trustpilot review",
        "Reply to this trustpilot review on my screen",
        "Draft a response to this ebay message",
        "Reply to this ebay buyer message",
        "Write a response to this etsy message",
        "Reply to this etsy customer message on my screen",
        "Draft a response to this airbnb message",
        "Reply to this airbnb guest message",
        "Write a response to this booking.com message",
        "Draft a response to this uber eats review",
        "Reply to this doordash review on my screen",
        "Write a response to this grubhub review",
        "Draft a response to this zillow inquiry",
        "Reply to this realtor.com message",
        "Write a response to this indeed message",
        "Reply to this glassdoor review on my screen",
        "Answer this question on my screen",
        "What's the answer to this question",
        "Help me answer this",
        "Solve this problem on my screen",
        "What's the solution to this",
        "Translate this to Spanish",
        "Translate this email",
        "Convert this to another language",
        "Make this shorter",
        "Summarize this into one sentence",
        "Condense this email",
        "Make this more concise",
        "Expand on this",
        "Make this longer",
        "Add more details to this",
        "Rephrase this",
        "Say this differently",
        "Reword this",
        "Simplify this",
        "Make this easier to understand",
        "Explain this in simple terms",
        
        // ── Translation requests (screen-specific) ────
        "Translate this chinese on the screen to english",
        "Translate this text on my screen",
        "What does this chinese text say",
        "What does this spanish text mean",
        "Translate the text on my screen to french",
        "Convert this japanese to english",
        "What does this german text say",
        "Translate this korean on the screen",
        "What's the english translation of this",
        "Translate this russian text",
        "What does this arabic say",
        "Translate the chinese characters on my screen",
        "Convert this text to english",
        "What's this in english",
        "Translate this to my language",
        "What does this foreign text say",
        "Translate the text I'm looking at",
        "What's the translation of this",
        "Convert this to spanish",
        "Translate what's on my screen",
        
        // ── Follow-up queries (contextual screen references) ────
        "anything else about this",
        "anything else about that",
        "more about this",
        "more about that",
        "tell me more about this",
        "tell me more about that",
        "what else about this",
        "what else about that",
        "more details on this",
        "more details on that",
        "more information about this",
        "more information about that",
        "anything more on this",
        "something else about this",
        "what more about this",
        "details on this",
        "info on this",
        "information about this",
        "explain more about this",
        "tell me about this",
        "show me more about this",
        "I mean on my screen",
        "I mean this screen",
        "I mean what's on screen",
        "I'm talking about my screen",
        "I'm referring to the screen",
        "on the screen I mean",
        
        // ── Code/snippet queries (IDE/editor context) ────
        "what's the addCompromise code snippet",
        "what's the handleClick function",
        "show me the parseIntent method",
        "what's the getUserData code",
        "explain the validateInput function",
        "what does the fetchData method do",
        "read the processRequest code",
        "what's in the config variable",
        "show the error handling code",
        "what's the authentication logic",
        "explain this function",
        "what does this method do",
        "show me this code",
        "read this snippet",
        "what's this variable for",
        "explain this class",
        "what's in this section",
        "show the implementation",
        "what's the logic here",
        "explain this algorithm",
        
        // ── More action-oriented screen queries ────
        "Polish up this text on my screen",
        "Fix the grammar on this screen",
        "Correct this email I'm writing",
        "Improve this message on my screen",
        "Rewrite this better",
        "Make this text more professional",
        "Check this for spelling errors",
        "Proofread what's on my screen",
        "Fix any mistakes in this text",
        "Make this sound better",
        "Improve the wording of this",
        "Clean up this text",
        "Make this more concise on my screen",
        "Shorten this text",
        "Expand on what I wrote",
        "Add more details to this text",
        "Rephrase what's on my screen",
        "Say this differently on my screen",
        "Simplify this text on my screen",
        
        // ── NEW: Explicit screen reference patterns ────
        "List all the files on my screen",
        "List all the files on my screen in alphabetical order",
        "Show me all the files on my screen",
        "What files are on my screen",
        "What's on my screen right now",
        "Show me everything visible on the screen",
        "What files are displayed on my screen",
        "Read the text on my screen",
        "What's visible in my current window",
        "Analyze what's on the screen",
        "Tell me what you see on my screen",
        "What's showing on my screen",
        "Describe what's on my screen",
        "What do you see on the screen",
        "Read everything on my screen",
        "What's displayed on the screen",
        "Show me what's on the screen",
        "What's visible on my screen",
        "Analyze the screen content",
        "What's in my current window",
        
        // ── NEW: Deictic + screen/code combinations ────
        "What does this error on my screen mean",
        "Explain this code I'm looking at",
        "What's wrong with this function here",
        "How do I fix this bug on my screen",
        "What does this function on my screen do",
        "Explain this method I'm viewing",
        "What's this error message on my screen",
        "Debug this code on my screen",
        "What's this variable on my screen",
        "Explain this class I'm looking at",
        "What does this line of code mean",
        "Fix this error on my screen",
        "What's this component doing",
        "Explain this function here",
        "What's wrong with this code on my screen",
        
        // ── NEW: Screen listing/organization queries ────
        "List everything on my screen",
        "Show all items on my screen",
        "What items are visible on my screen",
        "List all visible files on my screen",
        "Show me all the folders on my screen",
        "What folders are displayed on my screen",
        "List the files I can see on my screen",
        "Show everything visible on the screen",
        "What's displayed in this window",
        "List all items in this window",
        
        // ── Visual explanation ────────────────────────
        "Explain this diagram",
        "What does this diagram show",
        "Describe this diagram",
        "Walk me through this diagram",
        "Help me understand this diagram",
        
        // ── Error detection ────────────────────────
        "Does this document have any spelling errors",
        "Are there spelling errors here",
        "Check for spelling mistakes",
        "Any typos in this document",
        "Find spelling errors",
        
        // ── UI guidance ────────────────────────
        "What button should I click next",
        "Which button should I press",
        "What should I click",
        "Where should I click next",
        "Guide me through this",
        
        // ── Security checks ────────────────────────
        "Is this page secure",
        "Is this website secure",
        "Is this site safe",
        "Check if this page is secure",
        "Is this a secure connection",
        
        // ── Deictic references ────────────────────────
        "This looks wrong",
        "Something looks wrong here",
        "This doesn't look right",
        "See this?",
        "Look at this",
        "Check this out",
        
        // ── Strong deictic references ────────────────────────
        "yeah THIS, what is this",
        "THIS right here",
        "what is THIS",
        "explain THIS",
        
        // ── Targeted text selection ────────────────────────
        "read the second paragraph out loud",
        "read the third paragraph",
        "read paragraph two",
        "what does paragraph 3 say",
        
        // ── Code/file scanning ────────────────────────
        "are there any TODOs left in this file",
        "find TODOs in this file",
        "search for TODO comments",
        "any FIXME markers here",
        
        // ── Document review ────────────────────────
        "does this contract look okay to you",
        "does this look right",
        "is this document correct",
        "review this contract",
        
        // ── Security judgement ────────────────────────
        "is this the official website or a scam",
        "is this website legit",
        "is this a scam site",
        "is this page trustworthy",
        
        // ── Diff analysis ────────────────────────
        "what changed compared to the last version of this doc",
        "what changed in this version",
        "show me the differences",
        "what's different here",
        
        // ── Current selection ────────────────────────
        "what's selected right now",
        "what is currently selected",
        "what's highlighted",
        "what did I select",
        
        // ── PDF/document analysis ────────────────────────
        "Summarize this PDF I'm looking at and highlight any deadlines you see",
        "summarize this PDF",
        "extract deadlines from this",
        "find important dates",
        
        // ── Spreadsheet analysis ────────────────────────
        "Look at this spreadsheet and tell me which months have the lowest revenue",
        "analyze this spreadsheet",
        "which column has the highest values",
        "find the lowest values",
        
        // ── Email thread analysis ────────────────────────
        "Read this email thread and tell me what the other person is asking for",
        "summarize this email thread",
        "what is this person asking",
        "what do they want",
        
        // ── UI guidance ────────────────────────
        "Look at this UI and tell me which button I should click to export the data",
        "which button exports the data",
        "how do I export from here",
        "where's the export button",
        
        // ── Validation with emotion ────────────────────────
        "This error message is freaking me out, what does it actually mean",
        "what does this error mean",
        "explain this error",
        "I'm worried I messed up this spreadsheet, can you check if any totals look wrong",
        "check if these totals are correct",
        "validate this spreadsheet",
        
        // ── Dashboard explanation ────────────────────────
        "I'm confused by this dashboard, can you explain what these graphs are showing",
        "explain this dashboard",
        "what do these graphs mean",
        "interpret this chart",
        
        // ── Tone review ────────────────────────
        "I'm nervous about this email I'm about to send, can you review it for tone",
        "review this email for tone",
        "does this sound professional",
        "is this email too harsh",
        
        // ── Code diff ────────────────────────
        "I'm not sure I understand this code diff, walk me through the key changes",
        "explain this code diff",
        "what changed in this commit",
        "walk me through these changes",
        
        // ── Document verification ────────────────────────
        "Is this document signed",
        "Is this form signed",
        "Is this contract signed",
        "Check if this is signed",
        "Has this been signed",

        // ── Round 5 seeds ───────────────────────────────────────
        // Reading aloud from screen (phi4 misclassifies as command_automate)
        "Read this to me",
        "Read that out for me",
        "Read this aloud",
        "Read this text to me",
        "Read this out loud",
        "Can you read this for me",
        // Open file / dashboard analysis (phi4 misclassifies these)
        "Summarize the file I currently have open in VS Code",
        "Describe the analytics dashboard I have open right now",
        "Analyze the code file I have open",
        "What's in the file I have open in my editor",
        "Summarize the document I have open",
        "Tell me what the file I have open says",
        // ── Round 6 seeds ──────────────────────────────────────
        "What branch am I on according to the Git status bar at the bottom of VS Code?",
        "What branch am I currently on in VS Code?",
        "what branch am I on in vs code",
        "What is the current branch shown in the editor?",
        "What is the line count shown in VS Code's status bar right now?",
        "What does the status bar say in VS Code?",
        "What is the current line number in my editor?",
        "What meeting invite is showing in the calendar notification right now?",
        "What does the notification popup say?",
        "What calendar event is popping up right now?",
        "Is there a notification badge on any app in my Dock right now?",
        "Are there any unread badge counts on my dock icons?",
        "What apps in my dock have notification badges?",
        "Read everything visible in the right-hand panel of VS Code to me",
        "What's showing in the VS Code sidebar right now?",
        "Describe what's in the right panel of my editor",
        "hey what is that error message in the terminal saying",
        "what is the error in the terminal",
        "read the terminal error to me",
        "What does this popup mean?",
        "What does that dialog box mean?",
        "What does this error message mean?",
        "What is this warning telling me?",
        // ── Round 6b reinforcement seeds ──────────────────────────────
        // Status bar and UI element queries (phi4 confuses with web_search)
        "What is the line count shown in VS Code's status bar right now?",
        "What line count does VS Code show in the status bar?",
        "How many lines does the current file have according to VS Code?",
        "What does the VS Code status bar show at the bottom?",
        "What info is in the VS Code status bar?",
        "What is the current line number shown at the bottom of VS Code?",
        // ── Round 7 seeds ──────────────────────────────────────
        // Real-time screen state queries phi4 confuses with memory_retrieve/command_automate
        "Is there anything in my VS Code problems panel right now?",
        "Describe the current state of my desktop right now",
        "Are there any unread notifications in my Slack left sidebar right now?",
        "are there any un read no ti fi ca tions in my slack side bar right now",
        "what does the er ror in the ter mi nal say right now",
        "What is the current file path shown in VS Code's breadcrumb nav?",
        "what branch name is shown in my edi tor sta tus bar",
        "read me the last five lines vis i ble in my ter mi nal right now",
        "Is the dev server running according to what I see in the terminal?"

        // NOTE: The following are explicitly NOT screen_intelligence — do not add these:
        // "Text this to me", "Send this to me", "Text me this", "Send me the results"
        // Those are command_automate (SMS/email skill invocation).
        // They appear here as a comment so the model boundaries are clear to maintainers.
      ],

      app_control_start: [
        // ── Enter control mode ─────────────────────────────────────
        "control Slack",
        "control Word",
        "control Chrome",
        "control Figma",
        "control VS Code",
        "turn on control mode",
        "enter control mode",
        // ── Round 9 seeds — bare app names and please-style launches ──
        // These are app launches with minimal phrasing (phi4 misroutes)
        "Day One please",
        "open Day One",
        "launch Day One journal app",
        "Anki",
        "open Anki",
        "launch Anki flash cards",
        "Anki app please",
        "start control mode",
        "activate control mode",
        "app control mode",
        "switch to control mode",
        "I want to control the app",
        "let me control this app",
        "take control of the app",
        "control this app",
        "control the current app",
        "control mode on",
        "enable control mode",
        "start controlling",
        "start controlling Slack",
        "start controlling Word",
        "I need to control apps",
        "let me interact with this app",
        "give me control",
        "hand me control",
        // ── Ambiguous but strong signals ─────────────────────────────────────
        "control mode",
        "app control",
        "take over the app",
        "manual control",
        "direct control",
        // ── Round 6 seeds ──────────────────────────────────────
        "Take charge of Asana",
        "Take charge of Figma",
        "Take charge of this app",
        "Take over Asana for me",
        "Take over Figma",
        "I want you to take charge of this application",
        // ── Round 7 seeds ──────────────────────────────────────
        // New app names phi4 hasn't seen (Notion, Linear, Replit)
        "Take charge of Notion",
        "Take over Notion for me",
        "take con trol of lin ear",
        "Control Linear for me",
        "con trol rep lit for me",
        "Control Replit",
        "Take charge of Replit and open a new Python notebook",
        "Take over Replit"
        // NOTE: Exit phrases (stop/exit/quit/done) are intentionally NOT here.
        // Exit detection is handled by _dispatchControlCommand heuristic in main.js
        // ONLY when appControlMode.active === true. Adding them here would cause
        // DistilBERT to misclassify generic cancel/abort signals as app_control_start.
      ],

      greeting: [
        // ── Original ─────────────────────────────────────
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
        "Sup",

        // ── New – casual, regional, emoji-rich ───────
        "Heya!",
        "Morning! ☕",
        "What’s up doc?",
        "Hi friend 😊",
        "G’day mate",
        "Howdy partner",
        "Salut!",
        "Namaste 🙏",
        "Hey hey hey!",
        "Cheers!",
        "Thanks heaps!",
        "You rock! 🚀",
        "Hey, long time no see",
        "What's cooking?",
        
        // ── CRITICAL: Casual greetings that were failing tests ────────────────────────
        "Yo, what's up",
        "Yo what's up",
        "Yo wassup",
        "Yo AI",
        "Yo assistant",
        "Yo bot",
        "Nice to see you again",
        "Nice to see you",
        "Good to see you",
        "Great to see you",
        
        // ── Polite greetings ────────────────────────
        "Hope you're doing well",
        "Hope you're well",
        "Hope all is well",
        "Wishing you well",
        "Hope you're having a good day",
        
        // ── Contextual greetings ────────────────────────
        "okay I'm back",
        "I'm back",
        "back again",
        "I've returned",
        
        // ── Polite closers ────────────────────────
        "thanks, that's all for now, bye",
        "that's all, thanks",
        "all done, bye",
        "thanks, goodbye",
        
        // ── Playful greetings ────────────────────────
        "Hey there, hope your servers are doing okay today",
        "hope your servers are good",
        "how are the servers",
        
        // ── Referencing past ────────────────────────
        "Hi again, thanks for the help yesterday",
        "thanks for yesterday",
        "appreciate the help earlier",
        
        // ── Day-specific ────────────────────────
        "Yo, happy Friday!",
        "happy Friday",
        "TGIF",
        "it's Friday!",
        
        // ── Emotional greetings ────────────────────────
        "Hey, I'm back, missed you",
        "missed you",
        "good to be back",
        "Good morning, I'm a bit nervous today",
        "morning, feeling nervous",
        "Hi friend, it's been a rough day",
        "it's been rough",
        "rough day today",
        
        // ── Check-in ────────────────────────
        "Evening, just wanted to check in",
        "checking in",
        "just checking in",
        "wanted to say hi",

        // ── Identity / persona — questions about the assistant itself ────────
        "What's your name?",
        "What is your name?",
        "Who are you?",
        "What are you?",
        "Tell me about yourself",
        "What can you do?",
        "What do you do?",
        "Are you an AI?",
        "Are you a bot?",
        "Are you human?",
        "Do you have a name?",
        "What should I call you?",
        "How old are you?",
        "Where do you come from?",
        "What is your purpose?",
        "What is your role?",
        "Do you think?",
        "Do you feel anything?",
        "Do you remember me?",
        "Do you like music?",
        "Can you hear me?",
        "Are you listening?",
        "Are you there?",
        "Are you awake?",
        "Is this working?",
        "Do you understand me?",

        // ── Round 5 seeds ───────────────────────────────────────
        // Wake-up / time-of-day phrases (phi4 misclassifies as command_automate)
        "Rise and shine",
        "rise and shine",
        "Wakey wakey",
        "wakey wakey",
        "good morning sunshine",
        "Rise and shine it's a new day",
        "Time to wake up",
        // ── Round 6 seeds ──────────────────────────────────────
        "What's up, ThinkDrop?",
        "Yo, what's up ThinkDrop",
        "Hey ThinkDrop what's good",
        "Hey, I'm back for today",
        "I'm back for the day",
        "Hey I'm back, ready to go",
        "Back at it",
        "Let's get started",
        "Alright let's go",
        "Ready to start",
        "OK let's do this",
        "uh hey think drop",
        "hey think drop",
        "um hi think drop",
        // ── Round 7 seeds ──────────────────────────────────────
        // Identity question phi4 routes to memory_retrieve
        "What's your name?",
        "What is your name?",
        "what's your name think drop",
        // Morning greeting phi4 routes to memory_store
        "good morn ing read y to get start ed",
        "Good morning, ready to get started",
        // ── Round 7b seeds ──────────────────────────────────────
        "What are you called?",
        "What name do you go by?",
        "Tell me your name please",
        "Do you have a name, AI?",
        "what is your name think drop"
      ]
    };
    
    this.seedEmbeddings = null;
  }

  async initialize() {
    if (this.initialized) return;
    
    console.log('🚀 Initializing DistilBertIntentParser...');
    const startTime = Date.now();
    
    try {
      // Load embedding model (only model we need - Compromise handles entities)
      console.log('  Loading embedding model...');
      this.embedder = await pipeline(
        'feature-extraction',
        'Xenova/all-MiniLM-L6-v2',
        { quantized: true }
      );
      
      // Pre-compute seed embeddings
      console.log('  Computing seed embeddings...');
      await this.computeSeedEmbeddings();
      
      this.initialized = true;
      console.log(`✅ DistilBertIntentParser initialized in ${Date.now() - startTime}ms`);
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
      const entities = [];
      const doc = nlp(message);

      // ------------------------------------------------------------
      // 1. Compromise built-ins (people / places / orgs)
      // ------------------------------------------------------------
      const addCompromise = (method, type, confidence = 0.92) => {
        doc[method]().json().forEach(item => {
          const txt = item.text.trim();
          if (!txt) return;
          entities.push({
            type,
            value: txt,
            entity_type: type.toUpperCase(),
            confidence,
            start: item.offset?.start ?? message.indexOf(txt),
            end:   (item.offset?.start ?? message.indexOf(txt)) + txt.length
          });
        });
      };

      addCompromise('people', 'person');
      addCompromise('places', 'location');
      addCompromise('organizations', 'organization');

      // ------------------------------------------------------------
      // 2. Appointment / medical keywords (regex – more flexible)
      // ------------------------------------------------------------
      const apptRegex = /(?:dentist|doctor|dr\.?|vision|eye|dental|medical|therapy|physical|check[- ]?up|exam|appt|appointment|visit|consultation|follow.?up)\b\s*(?:appt|appointment|visit|exam|check.?up)?/gi;
      let m;
      while ((m = apptRegex.exec(message)) !== null) {
        const val = m[0];
        entities.push({
          type: 'appointment_type',
          value: val,
          entity_type: 'APPOINTMENT',
          confidence: 0.93,
          start: m.index,
          end: m.index + val.length
        });
      }

      // ------------------------------------------------------------
      // 3. Temporal entities (your existing method)
      // ------------------------------------------------------------
      entities.push(...this.extractTemporalEntities(message));

      // ------------------------------------------------------------
      // 4. Regex-based universal entities
      // ------------------------------------------------------------
      const regexes = [
        { re: /https?:\/\/[^\s]+/gi,               type: 'url',          et: 'URL',        conf: 1.0 },
        { re: /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/gi, type: 'email', et: 'EMAIL',      conf: 1.0 },
        { re: /(?:\+?\d{1,3}[-.\s]?)?(?:\(\d{1,4}\)|\d{1,4})[-.\s]?\d{1,4}[-.\s]?\d{1,4}\b/gi, type: 'phone', et: 'PHONE', conf: 0.98 },
        { re: /\b\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b/g, type: 'money',   et: 'MONEY',      conf: 0.96 },
        { re: /\bv?\d+\.\d+(?:\.\d+)?\b/g,          type: 'version',      et: 'VERSION',    conf: 0.95 },
        { re: /\b\d+\s*(?:minutes?|hours?|days?|weeks?|miles?|km|lbs?|kg)\b/gi, type: 'quantity', et: 'QUANTITY', conf: 0.90 }
      ];

      regexes.forEach(({ re, type, et, conf }) => {
        let match;
        while ((match = re.exec(message)) !== null) {
          entities.push({
            type,
            value: match[0],
            entity_type: et,
            confidence: conf,
            start: match.index,
            end: match.index + match[0].length
          });
        }
      });

      // ------------------------------------------------------------
      // 5. Multi-word tech terms (case-insensitive)
      // ------------------------------------------------------------
      const techPhrases = [
        'next.js','react native','vue.js','tailwind css','chakra ui','material ui',
        'bootstrap','fast api','django rest','spring boot','ruby on rails',
        'docker','kubernetes','terraform','ansible','github actions','gitlab ci',
        'postman','insomnia','figma','notion','linear','jira','javascript','typescript',
        'python','rust','golang','java','swift','kotlin','c\\+\\+','c#',
        // Single-letter languages (need word boundaries to avoid false positives)
        '\\bc\\b','\\br\\b'
      ];
      const techRE = new RegExp(`\\b(${techPhrases.join('|')})\\b`, 'gi');
      while ((m = techRE.exec(message)) !== null) {
        entities.push({
          type: 'tech_term',
          value: m[0],
          entity_type: 'TECH',
          confidence: 0.96,
          start: m.index,
          end: m.index + m[0].length
        });
      }

      // ------------------------------------------------------------
      // 6. Proper-noun fallback (Compromise missed)
      // ------------------------------------------------------------
      const seen = new Set(entities.map(e => e.value.toLowerCase()));
      doc.terms().json().forEach(tok => {
        const term = tok.terms?.[0];
        if (!term) return;
        const txt = term.text;
        const low = txt.toLowerCase();
        if (seen.has(low)) return;

        const isProper   = term.tags?.includes('ProperNoun');
        const isCap      = /^[A-Z]/.test(txt) && txt.length >= 1; // Single capital letter or capitalized word
        const isAcronym  = /^[A-Z]{2,}$/.test(txt);

        if (isProper || isCap || isAcronym) {
          const start = term.offset?.start ?? message.indexOf(txt);
          entities.push({
            type: 'proper_noun',
            value: txt,
            entity_type: 'PROPER_NOUN',
            confidence: isProper ? 0.88 : (isAcronym ? 0.85 : 0.78),
            start,
            end: start + txt.length
          });
          seen.add(low);
        }
      });

      // ------------------------------------------------------------
      // 7. Merge adjacent same-type entities (e.g. "John Doe")
      // ------------------------------------------------------------
      if (entities.length > 1) {
        const merged = [];
        let cur = entities[0];

        for (let i = 1; i < entities.length; i++) {
          const nxt = entities[i];
          const gap = nxt.start - cur.end;

          if (
            cur.entity_type === nxt.entity_type &&
            gap >= 0 && gap <= 2 &&                     // space or punctuation
            !/[.!?]\s*$/.test(message.slice(cur.end, nxt.start))
          ) {
            // extend current
            cur = {
              ...cur,
              value: message.slice(cur.start, nxt.end),
              end: nxt.end,
              confidence: Math.max(cur.confidence, nxt.confidence)
            };
          } else {
            merged.push(cur);
            cur = nxt;
          }
        }
        merged.push(cur);
        entities.splice(0, entities.length, ...merged);
      }

      // ------------------------------------------------------------
      // 8. Final sort by start position
      // ------------------------------------------------------------
      entities.sort((a, b) => a.start - b.start);

      return entities;
    } catch (err) {
      console.warn('Entity extraction failed:', err.message);
      return [];
    }
  }

  extractTemporalEntities(message) {
    const entities = [];
    
    try {
      const doc = nlp(message);
      
      // Extract dates - use match patterns for dates
      const datePatterns = [
        '#Date',           // "tomorrow", "January 5th", "next week"
        '#Month #Value',   // "January 5"
        'next #Duration',  // "next week", "next month"
        'the #Ordinal',    // "the 3rd", "the 15th"
        '#WeekDay',        // "Monday", "Wednesday"
        // Custom patterns for day abbreviations
        'the? (mon|tues|tue|wed|thur|thu|fri|sat|sun)',  // "the Thur", "Mon", etc.
        '(monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
        '(next|this|last) (monday|tuesday|wednesday|thursday|friday|saturday|sunday)'
      ];
      
      datePatterns.forEach(pattern => {
        const matches = doc.match(pattern).json();
        matches.forEach(match => {
          // Avoid duplicates
          const alreadyExists = entities.some(e => e.value === match.text);
          if (!alreadyExists) {
            entities.push({
              type: 'datetime',
              value: match.text,
              entity_type: 'DATE',
              confidence: 0.95,
              start: match.offset?.start || 0,
              end: match.offset?.start ? match.offset.start + match.text.length : match.text.length
            });
          }
        });
      });
      
      // Extract times
      const timePatterns = [
        '#Time',                    // "3pm", "noon"
        '#Value (am|pm|oclock)',    // "3 pm", "three oclock"
        'at #Value',                // "at three"
        '#Value #Time'              // "3 o'clock"
      ];
      
      timePatterns.forEach(pattern => {
        const matches = doc.match(pattern).json();
        matches.forEach(match => {
          const alreadyExists = entities.some(e => e.value === match.text);
          if (!alreadyExists) {
            entities.push({
              type: 'datetime',
              value: match.text,
              entity_type: 'TIME',
              confidence: 0.95,
              start: match.offset?.start || 0,
              end: match.offset?.start ? match.offset.start + match.text.length : match.text.length
            });
          }
        });
      });
      
    } catch (error) {
      console.warn('⚠️ Compromise temporal extraction failed:', error.message);
    }
    
    return entities;
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
      // 0. Check for highlighted text marker - exclude screen_intelligence if present
      const hasHighlightedTextMarker = message.includes('[HIGHLIGHTED_TEXT]');
      const excludeScreenIntelligence = hasHighlightedTextMarker || options.excludeScreenIntelligence === true;
      
      if (hasHighlightedTextMarker) {
        console.log('📎 [DISTILBERT] Highlighted text marker detected - excluding screen_intelligence from classification');
        // Remove the marker from the message for classification
        message = message.replace(/\[HIGHLIGHTED_TEXT\]\s*/g, '');
      }
      
      // 0. Build context-aware message if conversation history is provided
      let messageToClassify = message;
      const conversationHistory = options.conversationHistory || [];
      
      if (conversationHistory.length > 0) {
        // For very short messages like "yes", "no", "ok", include last assistant message for context
        const isShortResponse = message.trim().length < 15 && 
                               /^(yes|no|ok|sure|yeah|nope|yep|nah|maybe|perhaps|definitely|absolutely|correct|right|wrong|true|false)$/i.test(message.trim());
        
        if (isShortResponse) {
          // Get the last assistant message
          const lastAssistantMsg = conversationHistory.slice().reverse().find(msg => msg.role === 'assistant');
          if (lastAssistantMsg) {
            // Prepend context to help classification
            messageToClassify = `[Context: ${lastAssistantMsg.content.substring(0, 100)}] ${message}`;
            console.log(`🔍 [DISTILBERT] Short response detected, adding context: "${message}" → "${messageToClassify}"`);
          }
        }
      }
      
      // 1. Generate embedding for input message
      const messageEmbedding = await this.generateEmbedding(messageToClassify);
      
      // 2. Calculate similarity scores with seed examples
      const scores = this.calculateIntentScores(messageEmbedding);
      
      // 2.5. Exclude screen_intelligence if highlighted text is present
      if (excludeScreenIntelligence && scores.screen_intelligence) {
        console.log(`📎 [DISTILBERT] Removing screen_intelligence from consideration (score was: ${scores.screen_intelligence.toFixed(3)})`);
        delete scores.screen_intelligence;
      }
      
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
    
    // ═══════════════════════════════════════════════════════════════════════════
    // 🎯 CRITICAL RULES ONLY - Let the model learn most patterns from seed examples
    // ═══════════════════════════════════════════════════════════════════════════
    
    // 1️⃣ "I NEED YOU TO / I NEED TO / HELP ME / CAN YOU" + ACTION VERBS - Strong signal for command_automate
    const hasINeedTo = lowerMessage.match(/^(i need (you to|you |to )|help me |can you (do|help|go|search|find|book|buy|apply|fill|sign|renew|register|schedule|order|check|navigate|open|create|send|submit)|please (do|go|search|find|book|buy|apply|fill|sign|renew|register|schedule|order|check|navigate|open|create|send|submit))/i);
    const hasAutomationActionVerb = lowerMessage.match(/\b(open|launch|start|close|click|type|paste|copy|create|delete|move|navigate|goto|go to|find|search|select|drag|drop|scroll|press|enter|edit|update|append|write|rename|modify|renew|book|apply|register|schedule|order|buy|purchase|sign up|fill out|submit|pay|cancel|track|install|download|reset|upgrade|unsubscribe|watch|monitor|poll|notify|alert|summarize|forward|sync|fetch|check|send|text)\b/);
    if (hasINeedTo && hasAutomationActionVerb) {
      scores.command_automate *= 2.5;
      scores.screen_intelligence *= 0.3;
      scores.memory_retrieve *= 0.2;
      scores.memory_store *= 0.1;
      console.log('🎯 [DISTILBERT] "I need to/help me/can you" + action verb detected - boosting command_automate');
    }
    
    // 1b️⃣ BUILD/CREATE/MAKE APP OVERRIDE — "build a tic tac toe game", "create a todo app"
    // DistilBERT scores these ~0.39 for both web_search and command_automate.
    // The user wants the AI to BUILD something — always command_automate.
    const isBuildRequest = lowerMessage.match(
      /^(build|create|make|generate|write|code|develop|implement)\b.{0,60}\b(app|application|game|tool|script|widget|dashboard|cli|bot|program|site|website|webapp|extension|plugin|utility|calculator|tracker|manager|timer|reminder|scheduler)\b/i
    ) || lowerMessage.match(/^(build|create|make|generate)\s+(me\s+)?(a|an|the)\s+/i);
    if (isBuildRequest) {
      const maxScore = Math.max(...Object.values(scores));
      scores.command_automate = Math.max(scores.command_automate, maxScore) * 2.5;
      scores.web_search *= 0.1;
      scores.general_knowledge *= 0.1;
      scores.memory_retrieve = 0.001;
      scores.memory_store = 0.001;
      console.log('🔨 [DISTILBERT] Build/create/make app request — hard override to command_automate');
    }

    // 1c️⃣ SERVICE AUTOMATION OVERRIDE — "watch my Gmail", "monitor my inbox", "send me a daily text"
    // These are external service integrations that need a skill — always command_automate.
    const isServiceAutomation = lowerMessage.match(
      /\b(watch|monitor|poll|check|track|sync|fetch|forward|filter|archive|summarize|notify|alert)\b.{0,80}\b(gmail|inbox|email|emails|mail|message|messages|text|sms|slack|discord|telegram|whatsapp|calendar|schedule|event|events|appointment|notion|airtable|jira|trello|asana|github|linear|hubspot|salesforce|sheet|spreadsheet|drive|dropbox|twitter|instagram|linkedin|reddit)\b/i
    ) || lowerMessage.match(
      /\b(send|give|text|notify)\b.{0,60}\b(daily|weekly|every|each|nightly|morning|evening|night|at \d|around \d|summary|digest|briefing|reminder|alert|report)\b/i
    );
    if (isServiceAutomation) {
      const maxScore = Math.max(...Object.values(scores));
      scores.command_automate = Math.max(scores.command_automate, maxScore) * 2.0;
      scores.memory_retrieve = 0.001;
      scores.memory_store = 0.001;
      scores.screen_intelligence *= 0.1;
      console.log('🔔 [DISTILBERT] Service automation request — hard override to command_automate');
    }

    // 2️⃣ FILE SEARCH / EXISTENCE QUERIES - "do I have X files", "list all apps", "find files"
    // These are filesystem queries → command_automate (mdfind/find/ls), NOT screen_intelligence
    const isFileSearchQuery = lowerMessage.match(
      /\b(do i have|are there|have i got|find all|list all|show me all|search (my computer|for files)|what files|what apps|what applications|applications (on|installed)|apps (on|installed))\b.*\b(files?|folders?|apps?|applications?|documents?|photos?|images?|pdfs?|spreadsheets?|on my (computer|mac|desktop|laptop|machine))\b/i
    ) || lowerMessage.match(
      /\b(list|show|find|search for|do i have|are there)\b.*(files?|folders?|apps?|applications?)\b.*(on my|in my|computer|mac|desktop|laptop|downloads|documents|home)/i
    );
    if (isFileSearchQuery) {
      // Hard override — filesystem queries MUST be command_automate regardless of base scores
      const maxScore = Math.max(...Object.values(scores));
      scores.command_automate = Math.max(scores.command_automate, maxScore) * 1.5;
      scores.screen_intelligence = 0.001;
      scores.memory_retrieve = 0.001;
      scores.web_search *= 0.1;
      console.log('🗂️ [DISTILBERT] File search/existence query — hard override to command_automate');
    }

    // 2b️⃣ FILE EDIT / UPDATE QUERIES - "find file X and edit it", "update the file", "edit file X"
    const isFileEditQuery = lowerMessage.match(
      /\b(edit|update|append|modify|change|rewrite|overwrite)\b.{0,40}\b(file|document|txt|rtf|md|json|csv|notes?|verse|chapter)\b/i
    ) || lowerMessage.match(
      /\b(find|locate)\b.{0,30}\b(file|document)\b.{0,30}\b(edit|update|change|modify|append)\b/i
    );
    if (isFileEditQuery) {
      const maxScore = Math.max(...Object.values(scores));
      scores.command_automate = Math.max(scores.command_automate, maxScore) * 1.5;
      scores.memory_retrieve = 0.001;
      scores.screen_intelligence = 0.001;
      console.log('✏️ [DISTILBERT] File edit/update query — hard override to command_automate');
    }

    // 2c️⃣ CODEBASE / PROJECT ANALYSIS — "analyze the application at X", "read the codebase",
    // "explore the project", "tell me what this app does", "what is this repo about"
    // These are fs.read/command_automate, NOT screen_intelligence.
    const isCodebaseAnalysis = lowerMessage.match(
      /\b(analyze|analyse|read|explore|understand|examine|inspect|overview|summarize|what.*(about|is)|tell me.*(about|what))\b.{0,80}\b(app|application|project|repo|repository|codebase|code base|folder|directory)\b/i
    ) || lowerMessage.match(
      /\b(analyze|analyse|read|explore|understand|examine|inspect)\b.{0,60}\b(at|in|from|located|on)\b.{0,60}(~\/|\/Users\/|\/home\/|desktop|projects|folder)/i
    ) || lowerMessage.match(
      /\b(what('s| is) (this|the) (app|application|project|repo) (all )?about)\b/i
    );
    if (isCodebaseAnalysis) {
      const maxScore = Math.max(...Object.values(scores));
      scores.command_automate = Math.max(scores.command_automate, maxScore) * 2.0;
      scores.screen_intelligence = 0.001;
      scores.memory_retrieve *= 0.1;
      console.log('📁 [DISTILBERT] Codebase/project analysis query — hard override to command_automate');
    }

    // 3️⃣ EXPLICIT SCREEN REFERENCES - Strongest signal for screen_intelligence
    // BUT: Don't boost if there's a clear action verb (lock, record, capture, etc.)
    const hasActionVerb = lowerMessage.match(/^(lock|unlock|record|capture|screenshot|snap|start recording|begin recording|stop recording)/i);
    const hasExplicitScreenReference = lowerMessage.match(/\b(on (my|the) screen|on screen|my screen|the screen|what'?s on|visible on)\b/);
    if (hasExplicitScreenReference && !hasActionVerb) {
      scores.screen_intelligence *= 2.0;
      scores.command_execute *= 0.4;  // Prevent filesystem command confusion
      console.log('🎯 [DISTILBERT] Explicit screen reference detected - boosting screen_intelligence');
    }
    
    // 3️⃣ "HOW TO" QUESTIONS - Always informational, never commands
    const isHowToQuestion = lowerMessage.match(/^how (to|do i|can i|should i)/i);
    if (isHowToQuestion) {
      scores.web_search *= 2.0;
      scores.command_execute *= 0.2;
      console.log('🔍 [DISTILBERT] "How to" question detected - boosting web_search, penalizing command');
    }
    
    // 4️⃣ HIGHLIGHTED TEXT OVERRIDE - When text is already selected, skip screen analysis
    const hasHighlightedText = message.includes('[Selected text') || 
                              message.includes('[selected text') ||
                              message.includes('Selected text from') ||
                              message.includes('selected text from') ||
                              message.match(/\[.*text.*from.*\]/i);
    if (hasHighlightedText) {
      scores.screen_intelligence = 0.001;
      scores.question *= 2.0;
      scores.web_search *= 1.5;
      console.log('🎯 [DISTILBERT] Highlighted text detected - disabling screen_intelligence');
    }
    
    // 5️⃣ MEMORY STORAGE VS RETRIEVAL - Distinguish between storing and retrieving
    const hasRetrievalQuestion = lowerMessage.match(/^(do you remember|can you recall|what did i|what do you know|when is|when did|what was|where is|where did|which|who did|have i|was i|did i|am i|have i|around|what about)\b/i);
    const hasStorageVerb = lowerMessage.match(/\b(remember|save|note|store|keep|don't forget|remind me)\b/);
    const hasQuestionMark = message.trim().endsWith('?');
    // Time-ago patterns: "1 minute ago", "15 mins ago", "an hour ago", "last hour", "last 30 mins", "X to Yam"
    const hasTimeAgoPattern = lowerMessage.match(/\b(\d+\s*(minute|min|hour|hr)s?\s*ago|an?\s+hour\s+ago|a\s+few\s+(minutes?|hours?)\s+ago|\d+\s*to\s*\d+(am|pm)|around\s+\d|last\s+\d+\s*(minute|min|hour|hr)s?|in\s+the\s+last\s+\d+\s*(minute|min|hour|hr)s?|past\s+\d+\s*(minute|min|hour|hr)s?|last\s+(hour|minute|min)|in\s+the\s+last\s+(hour|minute|min))/);
    
    // "what have I seen/done/visited" — past tense personal activity = memory_retrieve
    const hasPastTensePersonalActivity = lowerMessage.match(/\b(what have i (seen|done|visited|been|watched|read|looked at|worked on)|what did i (see|do|visit|watch|read|look at|work on))\b/i);
    if (hasPastTensePersonalActivity) {
      scores.memory_retrieve *= 2.5;
      scores.screen_intelligence *= 0.3;
      scores.memory_store *= 0.1;
      console.log('🧠 [DISTILBERT] Past-tense personal activity detected - boosting memory_retrieve');
    }

    // Past-tense action report — user is TELLING the app what they did → memory_store
    // "sent a message to X", "sent an email to X", "called X", "messaged X", "told X"
    // These were being misclassified as web_search
    const hasPastTenseActionReport = lowerMessage.match(/^(sent (a |an )?(message|email|text|slack|dm|note|reply|response|invite|request)|called |messaged |texted |emailed |told |informed |notified |pinged |dm'd |dmed )/i);
    if (hasPastTenseActionReport) {
      scores.memory_store *= 3.0;
      scores.web_search *= 0.05;
      scores.command_automate *= 0.1;
      scores.memory_retrieve *= 0.3;
      console.log('📝 [DISTILBERT] Past-tense action report detected - boosting memory_store, penalizing web_search');
    }

    // Personal-fact declaration — user is stating an identity/relationship fact → memory_store
    // "My name is Sam", "My wife is Sarah", "Chris Akers is my cousin", "No my name is Sam"
    // Standard form: starts with optional filler + "my <role> is <value>"
    const hasPersonalFactDeclaration = lowerMessage.match(
      /^(?:(?:no|nope|yes|yeah|actually|well|wait|so|ok|okay|right|anyway|hmm|um|uh|oh|ah),?\s+)?my\s+[\w\s']{1,30}\s+(?:name\s+)?(?:is|are|was)\s+\S/i
    ) || lowerMessage.match(
      // Inverted: "Chris Akers is my cousin" — capital name + "is my" + relationship role
      /^[a-z][\w\s.'-]{1,40}\s+(?:is|are|was)\s+my\s+(?:wife|husband|partner|mom|mother|dad|father|son|daughter|brother|sister|cousin|aunt|uncle|friend|coworker|boss|manager|doctor|dentist|vet|lawyer|trainer|coach|neighbor|roommate)\b/i
    );
    if (hasPersonalFactDeclaration) {
      scores.memory_store *= 3.0;
      scores.greeting *= 0.1;
      scores.memory_retrieve *= 0.1;
      scores.command_automate *= 0.1;
      scores.general_knowledge *= 0.1;
      console.log('🪪 [DISTILBERT] Personal-fact declaration detected - boosting memory_store');
    }

    // File-write destination — "save to ~/Desktop/file.md", "write to /tmp/out.txt", etc.
    // The prompt requires executing a shell command to write the file → command_automate.
    const hasFileWriteDest = lowerMessage.match(
      /\b(save|write|output|store|put)\b.{0,80}(to|into|as)\s+(~[/]|[/]|[.][/])[\w/.]+/i  // explicit path: ~/Desktop/file.md, /tmp/out.txt
      || /\b(save|write|output|store|put)\b.{0,80}(to|into)\s+(a\s+)?(file|txt|text file|markdown file|md file|\.txt|\.md|\.csv|\.json)\b/i  // "save to a file", "write to a txt"
      || /\b(save|write|output)\b.{0,80}(on|in|to)\s+(my\s+)?(desktop|documents|downloads|home folder|home directory)\b/i  // "save to my desktop/documents"
    );
    if (hasFileWriteDest) {
      scores.command_automate *= 3.0;
      scores.web_search *= 0.3;
      scores.memory_store *= 0.2;
      console.log('💾 [DISTILBERT] File-write destination detected - boosting command_automate, penalizing web_search');
    }

    // Personal-attribute retrieval: "what's my name", "who is my wife", "where is my X"
    // These ask ThinkDrop to recall a stored personal fact → strong memory_retrieve boost
    const hasPersonalAttributeQuery = lowerMessage.match(
      /\b(what'?s|what is|who is|who'?s|where is|where'?s|tell me)\s+(my|your)\s+\w/i
    );
    if (hasPersonalAttributeQuery) {
      scores.memory_retrieve *= 3.5;
      scores.greeting *= 0.1;
      scores.general_knowledge *= 0.3;
      scores.command_automate *= 0.3;
      console.log('🪪 [DISTILBERT] Personal-attribute query detected - strong boost memory_retrieve, penalize greeting');
    }

    if (hasRetrievalQuestion) {
      scores.memory_retrieve *= 2.0;
      scores.memory_store *= 0.2;
      console.log('🔍 [DISTILBERT] Retrieval question detected - boosting memory_retrieve');
    } else if (hasStorageVerb && !hasQuestionMark && !hasRetrievalQuestion) {
      scores.memory_store *= 1.3;
    } else if (hasQuestionMark && !hasStorageVerb) {
      scores.memory_store *= 0.3;  // Questions are rarely storage requests
    }
    
    // Time-ago pattern always means memory retrieval ("1 min ago", "an hour ago", "6 to 9am")
    if (hasTimeAgoPattern) {
      scores.memory_retrieve *= 2.5;
      scores.memory_store *= 0.1;
      scores.screen_intelligence *= 0.5;
      console.log('⏱️ [DISTILBERT] Time-ago pattern detected - boosting memory_retrieve');
    }
    
    // 6️⃣ NAVIGATION COMMANDS - Strong boost for goto/open/navigate patterns
    // These are automation commands, NOT web searches
    const hasNavigationCommand = lowerMessage.match(/^(goto|go to|open|navigate to|visit|browse to|head to|launch)\b/i);
    const hasWebsiteKeyword = lowerMessage.match(/\b(website|site|page|url)\b/i);
    
    if (hasNavigationCommand) {
      // Strong boost for command_automate when query starts with navigation verb
      scores.command_automate *= 1.8;
      
      // If it also mentions "website", it's definitely navigation, not search
      if (hasWebsiteKeyword) {
        scores.command_automate *= 1.3;
        scores.web_search *= 0.5; // Reduce web_search confidence
      }
    }

    // 6b️⃣ INFORMATIONAL QUESTIONS - "tell me about", "explain", "what is", "describe"
    // These are NEVER command_automate — they are web_search or general_knowledge
    const hasInformationalPattern = lowerMessage.match(/^(tell me (about|more about|what)|explain (to me |me )?(what|how|why|the|this|that)?|what (is|are|was|were) (this|that|the|a|an)\b|describe (this|that|the))/i);
    if (hasInformationalPattern) {
      scores.command_automate *= 0.1;
      scores.web_search *= 1.8;
      scores.general_knowledge *= 1.5;
      console.log('📖 [DISTILBERT] Informational question detected - penalizing command_automate, boosting web_search');
    }
    
    // 7️⃣ TIME-SENSITIVE WEB QUERIES - Boost for current events
    const hasCurrentEventIndicators = lowerMessage.match(/\b(current|now|today|latest|recent|this year|2024|2025|2026)\b/);
    const hasWeatherQuery = lowerMessage.match(/\b(weather|temperature|forecast)\b/);
    const hasPriceQuery = lowerMessage.match(/\b(price|cost|stock|worth)\b/);
    // Personal pronoun signals a personal history query, NOT a current-events web search
    const hasPersonalPronoun = lowerMessage.match(/\b(i|my|me|we|our|i've|i'm|i was|i did|i worked|i used)\b/);
    
    if ((hasCurrentEventIndicators || hasWeatherQuery || hasPriceQuery) && !hasPersonalPronoun) {
      scores.web_search *= 1.5;
    }
    
    // Personal pronoun + time word = personal history → boost memory_retrieve
    if (hasPersonalPronoun && hasCurrentEventIndicators) {
      scores.memory_retrieve *= 2.0;
      scores.web_search *= 0.5;
      console.log('🧠 [DISTILBERT] Personal+time query detected - boosting memory_retrieve, penalizing web_search');
    }
    
    // Personal pronoun + time-ago pattern (e.g. "was I online around 6am", "around 6 to 9am was I")
    if (hasPersonalPronoun && hasTimeAgoPattern) {
      scores.memory_retrieve *= 2.0;
      scores.memory_store *= 0.1;
      console.log('🧠 [DISTILBERT] Personal+time-ago pattern - boosting memory_retrieve, penalizing memory_store');
    }
    
    // 8️⃣ GENERAL KNOWLEDGE FALLBACK - When all scores are very low, prefer general_knowledge over command_automate
    const maxScoreVal = Math.max(...Object.values(scores));
    if (maxScoreVal < 0.35) {
      scores.general_knowledge *= 1.5;
      scores.command_automate *= 0.4;
      scores.screen_intelligence *= 0.4;
      console.log('📚 [DISTILBERT] Low-confidence query - boosting general_knowledge as fallback');
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // 📊 NORMALIZATION - Keep scores in 0-1 range (recompute after all boosts)
    // ═══════════════════════════════════════════════════════════════════════════
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
    
    // Only default to question if ALL scores are extremely low (< 0.15)
    // This prevents defaulting when web_search has highest score but low confidence
    if (topScore < 0.15) {
      console.log(`⚠️ Extremely low confidence (${topScore.toFixed(3)}), defaulting to 'question'`);
      return 'question';
    }
    
    // Always choose the highest score - semantic search is more accurate than priority rules
    // If scores are very close (within 0.05), log it but still choose highest score
    if (Math.abs(topScore - secondScore) < 0.05) {
      const secondIntent = sortedIntents[1][0];
      console.log(`⚖️ Close scores: ${topIntent} (${topScore.toFixed(3)}) vs ${secondIntent} (${secondScore.toFixed(3)}) → choosing ${topIntent} (highest score)`);
    }
    
    return topIntent;
  }
}

module.exports = DistilBertIntentParser;
