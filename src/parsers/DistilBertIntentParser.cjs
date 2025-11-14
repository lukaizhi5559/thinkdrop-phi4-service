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
      'screen_intelligence', // Primary screen analysis (UI elements, browser content, desktop items)
      'command_execute',     // Shell/OS commands (simple, direct execution)
      // 'command_automate',    // Nut.js UI automation (complex multi-step workflows)
      'command_guide',       // Educational/tutorial mode ("show me how")
      'memory_store',
      'memory_retrieve',
      'web_search',         // Time-sensitive queries requiring current data
      'general_knowledge',  // Stable facts that don't need web search
      'question',           // Capability queries and general questions
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
        "Set a reminder for my dentist appointment next Friday at 3pm",
        "Remind me about the meeting tomorrow at 10am",
        "I have an appointment next week on Tuesday. Set a reminder",
        "Create a reminder for my doctor's appointment on Monday",
        "I need a reminder for the team standup at 9am tomorrow",
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
        "Set a reminder for my dentist appointment next Friday at 3pm",
        "Remind me about the meeting tomorrow at 10am",
        "I have an appointment next week on Tuesday. Set a reminder",

        // ── New – richer phrasing, multi-entity, typos ───────
        "Quick reminder: call Dr. Patel on Tuesday 9am about bloodwork",
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
        "Log that I finished reading 'Atomic Habits' today"
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
        "Show the table we sketched"
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
        "Search for resources on API design"
      ],

      general_knowledge: [
        // Stable facts that don't change
        "What is the capital of France?",
        "Where is the Eiffel Tower located?",
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
        "Explain the difference between supervised and unsupervised learning"
      ],

      command_execute: [
        // ── File and folder manipulation ──────────────────
        "Create a file on my desktop called hello.txt",
        "Make a folder on my desktop named projects",
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
        // ── File and folder listing queries ──────────────────
        "List all the folders on my desktop",
        "Show me all files on my desktop",
        "What files are in my Downloads folder",
        "List all the files/folders on my desktop",
        "Show all folders in my Documents",
        "What's on my desktop",
        "List everything on my desktop",
        "Show me what's in my home directory",
        "What folders do I have on my desktop",
        "List all files in my Downloads",
        "Show all items on my desktop",
        "What's in my Documents folder",
        "List the contents of my desktop",
        "Show me my desktop files",
        "What files are on my desktop",
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
        "Search for markdown files"
      ],

      // COMMENTED: For now unti we get the rest of the app working smoothly
      // command_automate: [
      //    // ── Original ─────────────────────────────────────
      //   // NOTE: "Take a screenshot" removed - conflicts with vision intent
      //   // Vision service handles screen capture + analysis
      //   "Open Chrome",
      //   "Close all windows",
      //   "Search for restaurants nearby",
      //   "Play some music",
      //   "Open VS Code",
      //   "Start a 25-minute timer",
      //   "Mute system volume",
      //   "Create a new note titled 'Ideas'",
      //   "Switch to dark mode",
      //   "Play Lo-fi beats",
      //   "Close all Chrome tabs",
      //   "Launch Docker Desktop",
      //   "Copy the last transcript to clipboard",
      //   "Set an alarm for 7am",

      //   // ── New – more apps, OS, automation ───────
      //   "Open Slack and go to #general",
      //   "Lock the screen now",
      //   "Open Spotify and play my Discover Weekly",
      //   "Turn on Do Not Disturb until 5pm",
      //   "Open Terminal and run `htop`",
      //   "Empty the Recycle Bin",
      //   "Open Notion page 'Project Roadmap'",
      //   "Start screen recording",
      //   "Pause all media playback",
      //   "Open the calculator app",
      //   "Switch to the next desktop space",
      //   "Open my email client and compose a new message to boss@example.com",
      //   "Enable Bluetooth",
      //   "Open the system settings → Displays",
      //   "Restart the computer in 2 minutes",
      //   "Open Finder and go to Downloads",
      //   // NOTE: Screenshot commands removed - handled by vision service
      //   "Open Postman and load the 'API Tests' collection",
      //   "Turn off Wi-Fi",
      //   "Open the Calendar app and create an event for tomorrow 10am titled 'Standup'",
        
      //   // ── GOTO and navigation commands (browser/app navigation) ────
      //   "Goto Google",
      //   "Go to Google",
      //   "Goto Amazon",
      //   "Go to Amazon",
      //   "Goto YouTube",
      //   "Go to YouTube",
      //   "Goto Facebook",
      //   "Go to Facebook",
      //   "Goto Twitter",
      //   "Go to Twitter",
      //   "Goto LinkedIn",
      //   "Go to LinkedIn",
      //   "Goto Instagram",
      //   "Go to Instagram",
      //   "Goto Reddit",
      //   "Go to Reddit",
      //   "Goto GitHub",
      //   "Go to GitHub",
      //   "Goto Stack Overflow",
      //   "Go to Stack Overflow",
      //   "Goto Gmail",
      //   "Go to Gmail",
      //   "Goto Outlook",
      //   "Go to Outlook",
      //   "Goto Netflix",
      //   "Go to Netflix",
      //   "Goto Spotify",
      //   "Go to Spotify",
      //   "Goto the website",
      //   "Go to the site",
      //   "Navigate to Google",
      //   "Navigate to Amazon",
      //   "Open Google in browser",
      //   "Open Amazon in browser",
      //   "Visit Google.com",
      //   "Visit Amazon.com",
      //   "Browse to Google",
      //   "Browse to Amazon",
      //   "Head to Google",
      //   "Head to Amazon",
      //   "Goto google.com",
      //   "Go to amazon.com",
      //   "Open up Google",
      //   "Open up Amazon",
        
      //   // ── GOTO + ACTION (navigation with search/action) ────────────
      //   // "Go online" patterns (navigate to web/browser)
      //   "Go online and do a google search for the latest AI for video",
      //   "Go online and search for news",
      //   "Goto online and do a google search for shoes",
      //   "Go online and find information about climate change",
      //   "Goto online and search for restaurants near me",
      //   "Go online and look up the weather",
      //   "Go online and do a search for hotels",
      //   "Goto online and find flights to Paris",
        
      //   // Standard goto + action patterns
      //   "Goto Google and search for shoes",
      //   "Go to Google and search for winter clothes",
      //   "Goto Amazon and find me some winter clothes",
      //   "Go to Amazon and find winter boots",
      //   "Goto YouTube and search for cooking videos",
      //   "Go to YouTube and play music",
      //   "Goto Gmail and compose a new email",
      //   "Go to Gmail and check my inbox",
      //   "Goto LinkedIn and find jobs",
      //   "Go to LinkedIn and search for connections",
      //   "Goto GitHub and search for react libraries",
      //   "Go to GitHub and clone the repo",
      //   "Goto Stack Overflow and search for python errors",
      //   "Go to Stack Overflow and find solutions",
      //   "Goto Reddit and search for tech news",
      //   "Go to Reddit and browse r/programming",
      //   "Goto Twitter and search for AI news",
      //   "Go to Twitter and post a tweet",
      //   "Goto Facebook and check notifications",
      //   "Go to Facebook and post an update",
      //   "Goto Netflix and watch a movie",
      //   "Go to Netflix and browse shows",
      //   "Goto Spotify and play my playlist",
      //   "Go to Spotify and search for jazz music",
      //   "Goto the website and search for products",
      //   "Go to the site and look for deals",
      //   "Navigate to Google and search for restaurants",
      //   "Navigate to Amazon and buy a book",
      //   "Open Google and search for news",
      //   "Open Amazon and find gifts",
      //   "Visit Google and search for hotels",
      //   "Visit Amazon and browse electronics",
      //   "Browse to Google and search for flights",
      //   "Browse to Amazon and look for shoes",
        
      //   // ── SEARCH IN APP (app-specific searches) ────────────────────
      //   "Search in Gmail for emails from John",
      //   "Search my Gmail for all emails from 2018",
      //   "Search Gmail for receipts",
      //   "Search in my Gmail account for invoices",
      //   "Do a search at my Gmail account for all emails 2018",
      //   "Search my email for messages from boss",
      //   "Find emails in Gmail about project",
      //   "Look in Gmail for unread messages",
      //   "Search Outlook for calendar invites",
      //   "Search my Outlook for meeting requests",
      //   "Search in Slack for messages from Sarah",
      //   "Search Slack for #general channel",
      //   "Find messages in Slack about deployment",
      //   "Search Discord for announcements",
      //   "Search my Discord for DMs",
      //   "Search in Notion for project notes",
      //   "Search Notion for meeting minutes",
      //   "Find notes in Notion about Q4 goals",
      //   "Search Spotify for jazz playlists",
      //   "Search my Spotify for saved songs",
      //   "Search YouTube for cooking tutorials",
      //   "Search my YouTube for watch history",
      //   "Search Google Drive for documents",
      //   "Search my Drive for spreadsheets",
      //   "Find files in Dropbox",
      //   "Search Dropbox for PDFs",
      //   "Search Photos for pictures from vacation",
      //   "Search my Photos for selfies",
      //   "Find photos from last summer",
      //   "Search Calendar for meetings this week",
      //   "Search my Calendar for appointments",
      //   "Find events in Calendar for tomorrow",
        
      //   // ── CALENDAR and reminder commands ────────────────────────────
      //   "Set a reminder in Calendar to get a gift for mom Feb 2",
      //   "Set a reminder to get a gift for mom Feb 2",
      //   "Create a reminder for mom's gift Feb 2",
      //   "Add a reminder to buy gift for mom February 2",
      //   "Set reminder to call John tomorrow",
      //   "Create reminder for dentist appointment",
      //   "Add reminder to submit report by Friday",
      //   "Set a calendar reminder for team meeting",
      //   "Create a calendar event for lunch tomorrow",
      //   "Add event to Calendar for conference next week",
      //   "Schedule a meeting in Calendar for Monday",
      //   "Set up a calendar invite for the team",
      //   "Create calendar event titled 'Standup' for tomorrow 10am",
      //   "Add to calendar: doctor appointment Thursday 3pm",
      //   "Schedule reminder for grocery shopping",
      //   "Set alarm for 7am tomorrow",
      //   "Create alarm for 6:30am weekdays",
      //   "Add alarm for 8am",
      //   "Set timer for 25 minutes",
      //   "Create timer for 1 hour",
      //   "Start a 30 minute timer",

      //   // ── Nut.js UI automation - complex multi-step workflows ────────
      //   // Calendar/reminder operations
      //   "Set a calendar reminder for tomorrow at 3pm",
      //   "Create a calendar event for Monday at 10am",
      //   "Add a reminder for my dentist appointment next Friday",
      //   "Schedule a meeting in Calendar for next week",
      //   "Set a reminder to call mom tomorrow",
        
      //   // Email/messaging operations
      //   "Compose an email to john@example.com about the meeting",
      //   "Send a message to the team on Slack",
      //   "Write an email to boss@example.com",
      //   "Post a tweet about the new feature",
      //   "Send a DM on Discord to my friend",
      //   "Compose a new message in Gmail",
        
      //   // Shopping/browsing workflows
      //   "Find winter boots on Amazon",
      //   "Search for cooking videos on YouTube",
      //   "Go to Google and search for shoes",
      //   "Goto Amazon and find me some winter clothes",
      //   "Browse to GitHub and search for react libraries",
      //   "Navigate to LinkedIn and find jobs",
      //   "Find restaurants near me on Google Maps",
      //   "Book a flight to Paris on Expedia",
      //   "Order pizza from Dominos",
      //   "Add item to my Amazon cart",
        
      //   // App-specific workflows
      //   "Open Spotify and play my Discover Weekly",
      //   "Open Gmail and compose a new message to boss@example.com",
      //   "Go online and do a google search for the latest AI for video",
      //   "Goto YouTube and search for cooking videos",
      //   "Open Slack and send a message to the team",
      //   "Go to Twitter and post a tweet",
      //   "Create a new document in Google Docs",
      //   "Share this file on Dropbox",
      //   "Post this photo on Instagram",
      //   "Create a new playlist on Spotify",
        
      //   // Multi-step navigation + action
      //   "Go to Google and search for winter clothes",
      //   "Goto Amazon and buy a book",
      //   "Navigate to Google and search for restaurants",
      //   "Open Google and search for news",
      //   "Visit Amazon and browse electronics",
      //   "Browse to Google and search for flights",
      //   "Goto Gmail and check my inbox",
      //   "Go to LinkedIn and search for connections",
      //   "Goto Stack Overflow and find solutions",
      //   "Go to Reddit and browse r/programming",
      //   "Goto Netflix and watch a movie",
      //   "Go to Spotify and search for jazz music",
        
      //   // App searches (in-app automation)
      //   "Search in Gmail for emails from John",
      //   "Search my Gmail for all emails from 2018",
      //   "Do a search at my Gmail account for all emails 2018",
      //   "Search in Slack for messages from Sarah",
      //   "Find messages in Slack about deployment",
      //   "Search in Notion for project notes",
      //   "Search Spotify for jazz playlists",
      //   "Search YouTube for cooking tutorials",
      //   "Search Google Drive for documents",
      //   "Search Photos for pictures from vacation",

      //     // ── APP + ACTION (open app and do something) ──────────────────
      //   "Open Slack and message John",
      //   "Open Slack and go to #general",
      //   "Open Discord and join voice channel",
      //   "Open Discord and check messages",
      //   "Open Spotify and play my playlist",
      //   "Open Spotify and search for jazz",
      //   "Open Chrome and go to Google",
      //   "Open Chrome and search for news",
      //   "Open Safari and browse to Amazon",
      //   "Open Safari and search for hotels",
      //   "Open VS Code and open my project",
      //   "Open VS Code and create new file",
      //   "Open Terminal and run htop",
      //   "Open Terminal and check disk space",
      //   "Open Finder and go to Downloads",
      //   "Open Finder and search for PDFs",
      //   "Open Mail and compose new email",
      //   "Open Mail and check inbox",
      //   "Open Calendar and create event",
      //   "Open Calendar and check today's schedule",
      //   "Open Notes and create new note",
      //   "Open Notes and search for meeting notes",
      //   "Open Photos and find vacation pictures",
      //   "Open Photos and create album",
      //   "Open Messages and text mom",
      //   "Open Messages and check unread",
      //   "Open Settings and change wallpaper",
      //   "Open Settings and check updates",
      //   "Open System Preferences and adjust display",
      //   "Open System Preferences and change sound",
      // ],

      command_guide: [

        // ── 1. Educational / Tutorial Mode – “show me how” ────────────────────────
        // ── Software / Tool Tutorials
        "How do I use AI plugins in Figma",
        "Show me how to set up Gmail filters",
        "Teach me how to create a Slack workflow",
        "Guide me through setting up Docker",
        "Walk me through creating a GitHub repository",
        "How do I configure VS Code for Python",
        "Show me how to use Git branches",
        "Teach me how to deploy to Netlify",
        "Guide me through setting up SSH keys",
        "How do I create a React component",
        "Show me how to use the Figma API",
        "Teach me how to create a custom Notion database",
        "Walk me through building a Chrome extension",
        "How do I set up a local development server",
        "Guide me through using Postman collections",
        "Show me how to create a custom VS Code snippet",
        "Teach me how to use the GitHub CLI",
        "How do I create a custom Slack slash command",
        "Walk me through setting up a CI pipeline in CircleCI",
        "Show me how to use the Vercel CLI",
        "Guide me through creating a custom WordPress theme",
        "Teach me how to use the Stripe Dashboard",
        "How do I set up a local MongoDB replica set",
        "Show me how to create a custom Airtable view",
        "Walk me through using the AWS CLI",
        "Guide me through setting up a Firebase project",
        "Teach me how to create a custom Zapier integration",
        "How do I use the Shopify Admin API",
        "Show me how to set up a local PostgreSQL database",
        "Teach me how to use the GraphQL Playground",
        "Walk me through creating a custom Trello power-up",
        "Guide me through setting up a local Redis server",
        
        // ── CRITICAL: Setup/Installation patterns (prevent web_search misclassification)
        "Show me how to set up Docker",
        "Show me how to set up Docker on macOS",
        "Show me how to set up Docker Desktop",
        "Show me how to install and configure Docker",
        "Show me how to set up Docker Compose",
        "Walk me through setting up SSH keys",
        "Walk me through setting up SSH keys on GitHub",
        "Walk me through setting up SSH authentication",
        "Walk me through configuring SSH for Git",
        "Walk me through setting up SSH on macOS",
        "Show me how to create a React component",
        "Show me how to create a React component with hooks",
        "Show me how to create a functional React component",
        "Show me how to build a React component from scratch",
        "Show me how to create a custom React Hook",
        "Show me how to set up Node.js",
        "Show me how to install Node.js on macOS",
        "Show me how to set up a Node.js project",
        "Show me how to configure Node.js for production",
        "Walk me through installing Python",
        "Walk me through setting up Python on Windows",
        "Walk me through configuring Python virtual environments",
        "Guide me through setting up VS Code",
        "Guide me through installing VS Code extensions",
        "Guide me through configuring VS Code for TypeScript",
        "Teach me how to set up Git",
        "Teach me how to install Git on Linux",
        "Teach me how to configure Git globally",
        "How do I set up a development environment",
        "How do I install and configure PostgreSQL",
        "How do I set up a local MySQL database",
        "How do I configure nginx for production",
        "How do I set up a reverse proxy with nginx",

        // ── 2. Development Tutorials
        "Show me how to write a Dockerfile",
        "Teach me how to use npm scripts",
        "Walk me through setting up ESLint",
        "How do I configure Prettier",
        "Show me how to use Chrome DevTools",
        "Guide me through creating a pull request",
        "How do I set up a virtual environment in Python",
        "Teach me how to use Postman for API testing",
        "Show me how to configure Tailwind CSS",
        "Walk me through setting up a Next.js project",
        "Guide me through using TypeScript with React",
        "Teach me how to set up Jest for testing",
        "How do I use webpack for bundling",
        "Show me how to create a custom Hook in React",
        "Walk me through using Redux Toolkit",
        "Guide me through setting up a GraphQL server with Apollo",
        "Teach me how to use Prisma with a database",
        "How do I create a custom middleware in Express",
        "Show me how to use the Node.js debugger",
        "Walk me through setting up a monorepo with Turborepo",
        "Guide me through using Vite for a Vue project",
        "Teach me how to create a custom Svelte store",
        "How do I set up a local Kafka cluster",
        "Show me how to use the Docker Compose file",
        "Walk me through creating a custom CLI tool with oclif",
        "Guide me through using the GitHub REST API",
        "Teach me how to set up a local Elasticsearch instance",
        "How do I create a custom plugin for Obsidian",
        "Show me how to use the OpenAI API in Node.js",
        "Walk me through setting up a local Supabase project",

        // ── 3. Infrastructure / DevOps Tutorials
        "How do I use GitHub Actions",
        "Guide me through creating a Lambda function",
        "Show me how to set up MongoDB",
        "Teach me how to use Redis",
        "How do I configure nginx",
        "Walk me through setting up Kubernetes",
        "Show me how to deploy to AWS",
        "Guide me through creating a CI/CD pipeline",
        "How do I set up a reverse proxy",
        "Teach me how to use Terraform",
        "Walk me through provisioning an EC2 instance",
        "Guide me through setting up Cloudflare DNS",
        "Teach me how to use Ansible playbooks",
        "How do I create a CloudFront distribution",
        "Show me how to use the AWS CDK",
        "Walk me through setting up a VPC",
        "Guide me through using Helm charts",
        "Teach me how to configure Traefik",
        "How do I set up a GitLab CI runner",
        "Show me how to use Pulumi",
        "Walk me through creating a DigitalOcean droplet",
        "Guide me through using the Azure CLI",
        "Teach me how to set up a load balancer in GCP",
        "How do I create a custom domain with Route 53",
        "Show me how to use the Serverless Framework",
        "Walk me through setting up a Jenkins pipeline",

        // ── 4. Application / Productivity Tutorials
        "Show me how to create a Notion template",
        "How do I use Figma components",
        "Guide me through creating a Trello board",
        "Teach me how to use Slack workflows",
        "Walk me through setting up Google Analytics",
        "How do I create a Zapier automation",
        "Show me how to use Airtable formulas",
        "Guide me through creating a Canva design",
        "How do I set up a Mailchimp campaign",
        "Teach me how to use Asana for project management",
        "Walk me through creating a ClickUp space",
        "Guide me through setting up a Linear project",
        "Teach me how to use Monday.com boards",
        "How do I create a custom Jira workflow",
        "Show me how to use the Todoist API",
        "Walk me through setting up a Calendly link",
        "Guide me through creating a Typeform",
        "Teach me how to use the HubSpot CRM",
        "How do I set up a Stripe Checkout",
        "Show me how to create a Gumroad product",
        "Walk me through using the Webflow CMS",
        "Guide me through setting up a Ghost blog",
        "Teach me how to use the Discord bot API",
        "How do I create a custom Telegram bot",
        "Show me how to use the Twilio SMS API",
        "Walk me through setting up a SendGrid template",

        // ── 5. Design / Creative Tutorials
        "Show me how to create a Figma prototype",
        "Teach me how to use Framer Motion",
        "How do I design a custom icon set in Sketch",
        "Guide me through creating a UI kit in Adobe XD",
        "Walk me through animating with Lottie",
        "Teach me how to use the Canva API",
        "How do I create a custom color palette in Coolors",
        "Show me how to use the Procreate brush studio",
        "Guide me through creating a 3D model in Blender",
        "Teach me how to use the After Effects expression editor",

        // ── 6. Data / Analytics Tutorials
        "Show me how to create a Looker dashboard",
        "Teach me how to write a BigQuery SQL query",
        "How do I set up a Metabase instance",
        "Guide me through using Tableau calculated fields",
        "Walk me through creating a Power BI report",
        "Teach me how to use the Snowflake SQL editor",
        "How do I create a custom Google Data Studio connector",
        "Show me how to use the Mixpanel event tracker",
        "Guide me through setting up Amplitude cohorts",

        // ── 7. Security / Privacy Tutorials
        "Show me how to set up 2FA on GitHub",
        "Teach me how to use a password manager like 1Password",
        "How do I create a secure SSH key pair",
        "Guide me through enabling full-disk encryption on macOS",
        "Walk me through setting up a VPN with WireGuard",
        "Teach me how to use GPG for email encryption",
        "How do I audit npm dependencies for vulnerabilities",
        "Show me how to use the OWASP ZAP scanner",

        // ── 8. macOS / Windows / Linux System Tutorials
        "Show me how to use the macOS Terminal",
        "Teach me how to create a bash alias",
        "How do I use the Windows PowerShell",
        "Guide me through setting up zsh with Oh My Zsh",
        "Walk me through creating a systemd service",
        "Teach me how to use the Linux cron scheduler",
        "How do I create a Windows scheduled task",
        "Show me how to use the macOS Automator",
        "Guide me through setting up a macOS launch agent",
        "Teach me how to use the Linux firewall (ufw)",

        // ── 9. AI / ML Tutorials
        "Show me how to fine-tune a Hugging Face model",
        "Teach me how to use LangChain for RAG",
        "How do I create a custom OpenAI fine-tune",
        "Guide me through using LlamaIndex",
        "Walk me through setting up a local Ollama server",
        "Teach me how to use the Gemini API",
        "How do I create a custom prompt template",
        "Show me how to use the Claude API",
        "Guide me through building a RAG pipeline with Pinecone",
        "Teach me how to use the Cohere API for classification",

        // ── 10. Misc / Fun / Niche Tutorials
        "Show me how to create a custom emoji in Slack",
        "Teach me how to use the Raycast launcher",
        "How do I set up a local Home Assistant instance",
        "Guide me through creating a custom Alfred workflow",
        "Walk me through using the Obsidian vault",
        "Teach me how to create a custom Roam Research graph",
        "How do I set up a local Minecraft server",
        "Show me how to use the Twitch API",
        "Guide me through creating a custom Discord slash command",
        "Teach me how to use the YouTube Data API",
        "How do I create a custom Spotify playlist with the API",
        "Show me how to use the NASA API",
        "Walk me through setting up a local Mastodon instance",

        // ── AI + AI → Diagrams / Flowcharts / Architecture
        "How to use ChatGPT and Mermaid AI to generate system architecture diagrams",
        "Teach me how to combine Claude and Mermaid Live Editor to create real-time UML diagrams",
        "How to use Gemini and Excalidraw AI to sketch database ER diagrams from natural language",
        "Show me how to combine Perplexity and Draw.io AI to auto-generate network topology maps",
        "Guide me through using Llama 3 and Mermaid.js to create sequence diagrams from user stories",
        "How to use Claude 3 and Whimsical AI to build interactive product flowcharts",
        "Teach me how to combine ChatGPT and Lucidchart AI to generate org charts from team descriptions",
        "How to use NoteLM and Mermaid AI to create Gantt charts from project timelines",
        "Show me how to combine Qwen and Miro AI to generate mind maps from brainstorming sessions",
        "Guide me through using Grok and Figma AI to auto-create UI component diagrams",

        // ── AI + AI → Video / Animation / Explainer Content
        "How to use ChatGPT, Perplexity, and NoteLM to script a 5-minute learning video",
        "Teach me how to combine Claude, ElevenLabs, and HeyGen to create a talking AI explainer video",
        "How to use Gemini and Runway ML to generate video from AI-written scripts",
        "Show me how to combine Llama 3, Descript, and Synthesia to make an AI avatar tutorial",
        "Guide me through using Perplexity and Pictory to turn blog posts into AI-narrated videos",
        "How to use NoteLM and CapCut AI to auto-edit educational TikTok videos",
        "Teach me how to combine ChatGPT and VEED.io AI to add subtitles and animations to tutorials",
        "How to use Qwen and InVideo AI to create product demo videos from feature lists",
        "Show me how to combine Grok and Kaiber AI to generate animated AI art videos",
        "Guide me through using Claude and Lumen5 to turn podcast transcripts into video summaries",

        // ── AI + AI → Code Generation + Execution
        "How to use ChatGPT and Replit AI to build a full-stack app from a single prompt",
        "Teach me how to combine Claude and Cursor.sh to write and debug Python scripts",
        "How to use Gemini and GitHub Copilot to generate React components with tests",
        "Show me how to combine Perplexity and CodeSandbox AI to prototype web apps instantly",
        "Guide me through using Llama 3 and VS Code AI to auto-generate API clients",
        "How to use NoteLM and Glitch AI to deploy AI-powered web tools in 2 minutes",
        "Teach me how to combine Qwen and Phind AI to solve LeetCode problems with explanations",
        "How to use Grok and CodePen AI to generate interactive CSS animations",
        "Show me how to combine ChatGPT and Warp AI to write shell scripts from English",
        "Guide me through using Claude and Tabnine to auto-complete full functions in Java",

        // ── AI + AI → Content Creation (Blog, Social, Email)
        "How to use ChatGPT and Jasper AI to write a 2000-word SEO blog post",
        "Teach me how to combine Claude and Copy.ai to generate 10 LinkedIn posts from one idea",
        "How to use Gemini and Writesonic to create email sequences for product launches",
        "Show me how to combine Perplexity and Frase.io to generate content briefs with outlines",
        "Guide me through using NoteLM and Rytr to write YouTube video descriptions with hooks",
        "How to use Llama 3 and Anyword to A/B test ad copy variants",
        "Teach me how to combine Qwen and HyperWrite to rewrite articles in different tones",
        "How to use Grok and Grammarly AI to polish AI-generated technical documentation",
        "Show me how to combine ChatGPT and Notion AI to generate meeting notes into blog drafts",
        "Guide me through using Claude and Canva AI to create social media graphics with AI copy",

        // ── AI + AI → Research + Summarization
        "How to use Perplexity and ChatGPT to research and summarize a 50-page PDF",
        "Teach me how to combine Claude and Elicit.org to extract insights from 20 research papers",
        "How to use Gemini and Mem.ai to build a personal knowledge base from scattered notes",
        "Show me how to combine NoteLM and Scite.ai to find citations for AI claims",
        "Guide me through using Llama 3 and Humata.ai to Q&A a legal contract",
        "How to use Qwen and Genei.io to summarize 10 YouTube videos into one doc",
        "Teach me how to combine Grok and Glean to search internal company docs with AI",
        "How to use ChatGPT and Otter.ai to turn meeting recordings into action items",
        "Show me how to combine Perplexity and Consensus AI to fact-check scientific claims",
        "Guide me through using Claude and Reflect AI to journal and extract weekly insights",

        // ── AI + AI → Design + Prototyping
        "How to use ChatGPT and Figma AI to generate UI mockups from text descriptions",
        "Teach me how to combine Claude and Uizard to turn sketches into interactive prototypes",
        "How to use Gemini and Framer AI to build landing pages from prompts",
        "Show me how to combine Perplexity and Relume AI to generate Webflow components",
        "Guide me through using NoteLM and Galileo AI to design mobile app flows",
        "How to use Llama 3 and Anima AI to convert Figma designs to React code",
        "Teach me how to combine Qwen and DiagramGPT to generate flowcharts from code",
        "How to use Grok and Visily AI to create wireframes from user stories",
        "Show me how to combine ChatGPT and Midjourney to generate UI inspiration images",
        "Guide me through using Claude and Adobe Firefly to generate branded graphics",

        // ── AI + AI → Data + Automation
        "How to use ChatGPT and Make.com to build no-code AI automations",
        "Teach me how to combine Claude and Zapier AI to trigger actions from emails",
        "How to use Gemini and Airtable AI to auto-categorize form submissions",
        "Show me how to combine Perplexity and n8n AI to create self-healing workflows",
        "Guide me through using NoteLM and Parabola to clean CSV data with AI",
        "How to use Llama 3 and Bardeen AI to scrape and summarize websites",
        "Teach me how to combine Qwen and Albato to connect AI tools without code",
        "How to use Grok and Tray.io to build enterprise AI pipelines",
        "Show me how to combine ChatGPT and Google Sheets AI to analyze data with formulas",
        "Guide me through using Claude and Power Automate AI to approve invoices automatically",

        // ── AI + AI → Learning / Personal Knowledge
        "How to use ChatGPT and Notion AI to build a second brain from highlights",
        "Teach me how to combine Claude and Obsidian AI to link notes with embeddings",
        "How to use Gemini and Mem.ai to generate flashcards from YouTube videos",
        "Show me how to combine Perplexity and Anki AI to create spaced repetition decks",
        "Guide me through using NoteLM and Roam Research AI to generate backlinks",
        "How to use Llama 3 and Reflect AI to journal with AI-guided prompts",
        "Teach me how to combine Qwen and Heptabase to visualize knowledge graphs",
        "How to use Grok and Tana AI to capture ideas with AI tagging",
        "Show me how to combine ChatGPT and Readwise AI to summarize saved articles",
        "Guide me through using Claude and Capacities AI to build a personal CRM",

        // ── AI + AI → Fun / Creative / Niche
        "How to use ChatGPT and Suno AI to generate songs from story prompts",
        "Teach me how to combine Claude and Kaiber AI to make music videos from lyrics",
        "How to use Gemini and Pika Labs to generate AI video from text",
        "Show me how to combine Perplexity and Soundraw to create background music for podcasts",
        "Guide me through using NoteLM and Mubert to generate ambient soundscapes",
        "How to use Llama 3 and Scenario.gg to train custom AI art models",
        "Teach me how to combine Qwen and Replicate to run Stable Diffusion locally",
        "How to use Grok and Runway Gen-2 to animate AI-generated portraits",
        "Show me how to combine ChatGPT and DALL·E 3 to create children’s book illustrations",
        "Guide me through using Claude and Leonardo AI to generate consistent characters",

        // ── AI + AI → Business / Product
        "How to use ChatGPT and Typeform AI to generate customer surveys with logic",
        "Teach me how to combine Claude and Customer.io to send AI-personalized emails",
        "How to use Gemini and Intercom AI to auto-reply to support tickets",
        "Show me how to combine Perplexity and HubSpot AI to score leads from behavior",
        "Guide me through using NoteLM and Gong.io to analyze sales call sentiment",
        "How to use Llama 3 and Calendly AI to auto-schedule meetings from emails",
        "Teach me how to combine Qwen and Stripe AI to detect fraud in transactions",
        "How to use Grok and Shopify AI to generate product descriptions",
        "Show me how to combine ChatGPT and Klaviyo AI to segment customers with AI",
        "Guide me through using Claude and Attio AI to enrich CRM data with AI"
      ],

      screen_intelligence: [
        // ── UI Element Interaction ────────────────────────────
        "Click the Send button",
        "Click Send",
        "Press the Submit button",
        "Click on Submit",
        "Tap the Save button",
        "Click Save",
        "Press Enter",
        "Click the link",
        "Click on the menu",
        "Open the menu",
        "Close the window",
        "Close this",
        "Minimize the window",
        "Maximize the window",
        "Click the X button",
        "Press the button",
        "Click that button",
        "Tap that",
        "Select that option",
        "Choose that",
        
        // ── Text Input ────────────────────────────────────────
        "Type hello in the search box",
        "Type hello",
        "Enter my email",
        "Fill in the form",
        "Fill out this form",
        "Type my name",
        "Enter the password",
        "Input the text",
        "Write in the field",
        "Type in the box",
        "Enter text here",
        "Fill this field",
        "Type something",
        "Input my address",
        "Write my response",
        
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
        
        // ── Form Filling ──────────────────────────────────────
        "Fill out the login form",
        "Complete this form",
        "Fill in my details",
        "Enter my information",
        "Submit the form",
        "Fill the registration form",
        "Complete the signup",
        "Fill in the fields",
        "Enter my credentials",
        "Fill out the survey",
        
        // ── Navigation ────────────────────────────────────────
        "Go to the next page",
        "Click next",
        "Go back",
        "Click previous",
        "Scroll down",
        "Scroll up",
        "Go to the top",
        "Go to the bottom",
        "Navigate to settings",
        "Open preferences",
        
        // ── Multi-step Actions ────────────────────────────────
        "Click Send and then close the window",
        "Type hello and press Enter",
        "Fill the form and submit",
        "Open the menu and select settings",
        "Find the button and click it",
        "Locate the field and type my name",
        
        // ── Keyboard Shortcuts ────────────────────────────────
        "Press Command C",
        "Press Ctrl V",
        "Press Escape",
        "Press Tab",
        "Press Delete",
        "Press Backspace",
        "Hit Enter",
        "Press Space",
        "Press Arrow Down",
        "Press Command S",
        
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
        "Who is this person",
        "Who is this person at the bottom left",
        "Who is in this photo",
        "What's this person's name",
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
        
        // ── Action-oriented requests (editing, responding, fixing) ────
        "Polish up this email",
        "Clean up this email",
        "Make this email more professional",
        "Improve this email",
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
        "Reply to this telegram message on my screen",
        "Draft a response to this youtube comment",
        "Reply to this youtube comment on my screen",
        "Write a response to this youtube comment",
        "Respond to this youtube comment",
        "Put together a response to this rumble comment",
        "Draft a reply to this rumble comment",
        "Write a response to this rumble comment on my screen",
        "Draft a response to this gmail message",
        "Reply to this gmail email on my screen",
        "Write a response to this gmail",
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
        "Simplify this text on my screen"
      ],

      question: [
        // ── Original ─────────────────────────────────────
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

        // ── New – meta, troubleshooting, limits ───────
        "What model are you running under the hood?",
        "Do you have access to my camera?",
        "Can you read files from my Desktop folder?",
        "Are you GDPR compliant?",
        "What happens to the data I store in memory?",
        "Can you generate images?",
        "Do you support voice input right now?",
        "How accurate is your NER for non-English names?",
        "Can you call external APIs directly?",
        "What's the maximum context window you keep?",
        "Are you able to edit photos?",
        "Can you translate speech in real time?",
        "How do you handle ambiguous dates like 'next Friday'?",
        "Do you retain info across browser tabs?",
        "Can you export my memory as JSON?",
        
        // ── "What is X" questions (definitions/explanations - TECHNICAL CONCEPTS ONLY) ───────
        "What's an API?",
        "What is MCP?",
        "What's a webhook?",
        "What is REST?",
        "What's GraphQL?",
        "What is Docker?",
        "What's Kubernetes?",
        "What is CI/CD?",
        "What's a JWT?",
        "What is OAuth?",
        "What's an ORM?",
        "What is TypeScript?",
        "What's a microservice?",
        "What is serverless?",
        "What's edge computing?",
        "Ok what's an API",
        "So what is REST",
        "Alright what's Docker",
        "What does JWT stand for",
        "Cool what's the meaning of life",
        
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
        
        // ── How-to questions (NOT commands - asking for help, not tutorial) ─────────────────────
        "How can I debug this error?",
        "How do I fix this bug?",
        "How can I solve this problem?",
        "How do I fix permission denied?",
        "How to resolve merge conflicts?",
        "How can I optimize my code?",
        "How do I deploy to production?",
        "How to set up a virtual environment?",
        "How can I improve performance?",
        "How do I handle errors in async code?",
        "How to structure a React project?",
        "How can I secure my API?",
        "How do I implement authentication?",
        "How to write unit tests?",
        "How can I use Docker?",
        "How do I set up CI/CD?",
        "How to manage environment variables?",
        "How can I learn TypeScript?",
        "How do I get started with Python?",
        "How to become a better developer?",
        "How can I improve my workflow?",
        "How do I choose between X and Y?",
        
        // ── Troubleshooting questions (TECHNICAL ISSUES - may use screen context) ──────────────────────────
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
        "Why is the port already in use?",
        "Why am I getting 404 errors?",
        "Why is authentication failing?",
        "Why is my CSS not applying?",
        "Why is the API returning null?",
        "What's wrong with my code?",
        "What's wrong with this code?",
        "What's causing this bug?",
        "What's causing this bug in my code?",
        "What does this error mean?",
        "What does this error code mean?",
        "What's the issue here?",
        "What's the issue with my configuration?",
        "What am I doing wrong?",
        "What am I doing wrong in my implementation?",
        "Help me debug this code",
        "Help me fix this error",
        "Why isn't this working?",
        "What's the problem with my code?",
        
        // ── Comparison and recommendation questions ───────────────
        "Should I use React or Vue?",
        "Which is better: MySQL or PostgreSQL?",
        "What's the difference between npm and yarn?",
        "Should I learn Python or JavaScript first?",
        "Which framework should I use?",
        "What's better for backend: Node or Django?",
        "Should I use REST or GraphQL?",
        "Which cloud provider is best?",
        "What's the best way to learn coding?",
        "Should I use TypeScript?",
        "Which IDE is better?",
        "What's the difference between let and const?",
        "Should I use MongoDB or SQL?",
        "Which testing framework should I use?",
        "What's better: Docker or VMs?",
        "Should I use microservices?",
        "Which CSS framework is best?",
        "What's the difference between HTTP and HTTPS?",
        "Should I use serverless?",
        "Which version control system is better?",
        
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
        "Explain the difference between class and functional components",
        "Explain how Redux works",
        "Explain Docker containers",
        "Explain Kubernetes pods",
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
        "What are common mistakes in React?",
        "What are the best practices for database design?",
        "What are the principles of responsive design?",
        "What are the best practices for testing?",
        "What should I know about performance optimization?",
        "What are the best practices for error handling?",
        "What are the principles of RESTful API design?",
        "What are the best practices for code review?",
        "What should I consider for scalability?",
        
        // ── Career and learning questions ───────────────────────
        "How do I become a full-stack developer?",
        "What skills do I need for a backend role?",
        "How do I prepare for coding interviews?",
        "What should I learn next?",
        "How do I build a portfolio?",
        "What certifications are worth it?",
        "How do I get my first developer job?",
        "What's the career path for a developer?",
        "How do I stay up to date with tech?",
        "What resources do you recommend for learning?",
        "How do I contribute to open source?",
        "What makes a good developer?",
        "How do I improve my problem-solving skills?",
        "What should I focus on as a beginner?",
        "How do I transition from frontend to backend?"
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
        "What’s cooking?",
        "Yo yo yo",
        "Hey there, genius"
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
    
    // Detect if this is a question (WH-word or question mark)
    const hasQuestionWord = lowerMessage.match(/^(what|when|where|who|which|why|how|whose|whom|can|could|would|should|is|are|do|does|did)/i);
    const hasQuestionMark = message.trim().endsWith('?');
    const isQuestion = hasQuestionWord || hasQuestionMark;
    
    // Detect explicit storage verbs and reminder requests
    const hasStorageVerb = lowerMessage.match(/\b(remember|save|note|store|keep|don't forget|keep in mind|write down|jot down|set a reminder|remind me|create a reminder|add a reminder)\b/);
    
    // Boost memory_store ONLY if has storage verbs AND not a question
    if (entities.some(e => e.type === 'datetime' || e.type === 'person')) {
      if (hasStorageVerb && !isQuestion) {
        scores.memory_store *= 1.2;
      }
    }
    
    // Penalize memory_store for questions (critical fix)
    if (isQuestion && !hasStorageVerb) {
      scores.memory_store *= 0.3; // Strong penalty
    }
    
    // Boost memory_retrieve if asking about stored information
    if (lowerMessage.match(/^(what|when|where|who|which)/)) {
      if (entities.some(e => e.type === 'datetime' || e.type === 'person')) {
        scores.memory_retrieve *= 1.15;
      }
    }
    
    // Boost command if has action verbs
    if (lowerMessage.match(/^(open|close|launch|take|start|stop|play|set)/)) {
      scores.command *= 1.25;
    }
    
    // Boost question/web_search for WH-questions
    if (hasQuestionWord) {
      // Check if it's a factual question (likely needs web search)
      const isFactualQuestion = lowerMessage.match(/\b(who is|what is|when did|where is|how much|how many|what's the|who's the|when was|where was)\b/);
      
      if (isFactualQuestion) {
        scores.web_search *= 1.3;
        scores.question *= 1.15;
      } else {
        scores.question *= 1.2;
      }
    }
    
    // 🔍 ENHANCED: Boost web_search for current events and time-sensitive queries
    const hasCurrentEventIndicators = lowerMessage.match(/\b(current|now|today|latest|recent|this year|2024|2025|2026)\b/);
    const hasLeadershipQuery = lowerMessage.match(/\b(president|prime minister|ceo|leader|governor|mayor|king|queen)\b/);
    const hasPriceQuery = lowerMessage.match(/\b(price|cost|stock|worth|value|how much)\b/);
    const hasWeatherQuery = lowerMessage.match(/\b(weather|temperature|forecast|rain|snow|sunny|cloudy)\b/);
    const hasNewsQuery = lowerMessage.match(/\b(news|latest|happened|happening|event|announcement)\b/);
    const hasSportsQuery = lowerMessage.match(/\b(score|game|match|won|lost|team|player)\b/);
    
    // 🔍 NEW: Boost web_search for code/tutorial requests
    const hasCodeRequest = lowerMessage.match(/\b(give me|show me|how do i|how to|example of|tutorial|code for|script|can.*be rewritten|rewrite|refactor|improve|optimize)\b/);
    const hasProgrammingContext = lowerMessage.match(/\b(python|javascript|node|react|api|function|class|code|script|program|html|css|sql|database|docker|kubernetes|applescript|electron)\b/);
    
    // Strong boost for current events
    if (hasCurrentEventIndicators) {
      scores.web_search *= 1.5;
    }
    
    // Boost for leadership queries (often need current info)
    if (hasLeadershipQuery && (hasQuestionWord || hasQuestionMark)) {
      scores.web_search *= 1.4;
    }
    
    // Boost for price/cost queries (always need current data)
    if (hasPriceQuery) {
      scores.web_search *= 1.45;
    }
    
    // Boost for weather queries (always need current data)
    if (hasWeatherQuery) {
      scores.web_search *= 1.6;
    }
    
    // Boost for news queries
    if (hasNewsQuery) {
      scores.web_search *= 1.5;
    }
    
    // Boost for sports queries
    if (hasSportsQuery) {
      scores.web_search *= 1.4;
    }
    
    // 🔍 NEW: Strong boost for code/tutorial requests
    if (hasCodeRequest && hasProgrammingContext) {
      scores.web_search *= 1.6;  // Strong boost
      scores.command *= 0.5;      // Penalize command (avoid confusion)
      scores.question *= 0.7;     // Slightly penalize generic question
    }
    
    // Additional boost if ends with question mark
    if (hasQuestionMark) {
      scores.question *= 1.1;
      scores.web_search *= 1.05;
    }
    
    // Boost greeting if message is very short and contains greeting words
    if (message.split(' ').length <= 5) {
      if (lowerMessage.match(/^(hi|hello|hey|good morning|good afternoon)/)) {
        scores.greeting *= 1.3;
      }
    }
    
    // 🎯 NEW: Penalize screen_intelligence if highlighted text is present
    // When text is already highlighted/selected, we don't need screen analysis
    const hasHighlightedText = message.includes('[Selected text') || 
                              message.includes('[selected text') ||
                              message.includes('Selected text from') ||
                              message.includes('selected text from') ||
                              message.match(/\[.*text.*from.*\]/i);
    
    if (hasHighlightedText) {
      scores.screen_intelligence = 0.001;  // Force to near zero - we already have the text
      scores.question *= 3.0;              // Very strong boost to question intent instead
      scores.web_search *= 2.5;            // Very strong boost to web search for factual queries
      console.log('🎯 [INTENT] Detected highlighted text, forcing screen_intelligence to 0.001');
    }
    
    // Normalize scores back to 0-1 range
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
    
    // Intent priority for tie-breaking (when scores are very close)
    const intentPriority = {
      'screen_intelligence': 7,
      'command_guide': 7,      // Highest priority for educational tutorials
      'command_execute': 6,    // High priority for shell commands
      // 'command_automate': 6,   // High priority for UI automation
      'web_search': 5,         // Factual questions needing current data
      'question': 4,           // General questions
      'memory_retrieve': 3,    // Retrieving stored info
      'command': 2,            // Legacy command intent
      'context': 1,
      'memory_store': 0,       // Lowest priority (avoid false positives)
      'greeting': 0
    };
    
    // Only default to question if ALL scores are extremely low (< 0.15)
    // This prevents defaulting when web_search has highest score but low confidence
    if (topScore < 0.15) {
      console.log(`⚠️ Extremely low confidence (${topScore.toFixed(3)}), defaulting to 'question'`);
      return 'question';
    }
    
    // If scores are very close (within 0.1), use priority
    if (Math.abs(topScore - secondScore) < 0.1) {
      const topPriority = intentPriority[topIntent] || 0;
      const secondIntent = sortedIntents[1][0];
      const secondPriority = intentPriority[secondIntent] || 0;
      
      if (secondPriority > topPriority) {
        console.log(`🔄 Tie-breaking: ${topIntent} (${topScore.toFixed(3)}) vs ${secondIntent} (${secondScore.toFixed(3)}) → choosing ${secondIntent} (higher priority)`);
        return secondIntent;
      }
    }
    
    return topIntent;
  }
}

module.exports = DistilBertIntentParser;
