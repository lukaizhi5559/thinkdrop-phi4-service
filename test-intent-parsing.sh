#!/bin/bash

# Intent Parsing Test Suite
# Tests 100 diverse messages to verify intent classification

API_KEY="MyY6oYM3dO9-6ufn67xzyUvHQT-lunYVDaVLDDB7ZEg"
BASE_URL="http://localhost:3003"
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test function
test_intent() {
    local message="$1"
    local expected_intent="$2"
    local test_num="$3"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    # Make API call
    response=$(curl -s -X POST "$BASE_URL/intent.parse" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $API_KEY" \
        -d "{
            \"version\": \"mcp.v1\",
            \"service\": \"phi4\",
            \"action\": \"intent.parse\",
            \"requestId\": \"test-$test_num\",
            \"payload\": {
                \"message\": \"$message\"
            }
        }")
    
    # Extract intent from response
    actual_intent=$(echo "$response" | grep -o '"intent":"[^"]*"' | cut -d'"' -f4)
    confidence=$(echo "$response" | grep -o '"confidence":[0-9.]*' | cut -d':' -f2)
    elapsed=$(echo "$response" | grep -o '"elapsedMs":[0-9]*' | cut -d':' -f2)
    
    # Check if test passed
    if [ "$actual_intent" = "$expected_intent" ]; then
        PASSED_TESTS=$((PASSED_TESTS + 1))
        echo -e "${GREEN}✓${NC} Test $test_num: PASS - \"$message\" → $actual_intent (conf: $confidence, ${elapsed}ms)"
    else
        FAILED_TESTS=$((FAILED_TESTS + 1))
        echo -e "${RED}✗${NC} Test $test_num: FAIL - \"$message\""
        echo -e "   Expected: $expected_intent, Got: $actual_intent (conf: $confidence)"
    fi
}

echo "=========================================="
echo "Intent Parsing Test Suite - 100 Tests"
echo "=========================================="
echo ""

# ══════════════════════════════════════════════════
# Memory Store Tests (20 tests)
# ══════════════════════════════════════════════════
echo -e "${YELLOW}Memory Store Tests (20)${NC}"
test_intent "Save my gym locker combo: 14-28-03" "memory_store" 1
test_intent "I take 20mg of Lipitor every night, remember that" "memory_store" 2
test_intent "Note that my car is parked in spot B-17" "memory_store" 3
test_intent "Store this: my blood type is O-negative" "memory_store" 4
test_intent "Keep in mind I'm allergic to shellfish" "memory_store" 5
test_intent "Log that I finished reading Atomic Habits today" "memory_store" 6
test_intent "I have a flight on December 15th at 6am" "memory_store" 7
test_intent "Record this: license plate 7XYZ123 expires 2026-08-31" "memory_store" 8
test_intent "Jot down that I owe Sarah forty dollars for dinner" "memory_store" 9
test_intent "Save the vet appointment for Max on 11/18 at 3:45pm" "memory_store" 10
test_intent "From now on use Celsius unless I say otherwise" "memory_store" 11
test_intent "Remember my manager's name is Sarah and she's in London" "memory_store" 12
test_intent "I'm starting a new workout plan next Monday" "memory_store" 13
test_intent "Add parking garage level 3 row D to my notes" "memory_store" 14
test_intent "Please remember I like short concise answers" "memory_store" 15
test_intent "Store this API token: sk_live_abc123xyz" "memory_store" 16
test_intent "My sister's name is Emily, remember that" "memory_store" 17
test_intent "I'm trying to cut down on sugar for the next three months" "memory_store" 18
test_intent "Save my passport number for later" "memory_store" 19
test_intent "Consider Friday my cheat day, remember" "memory_store" 20

# ══════════════════════════════════════════════════
# Memory Retrieve Tests (20 tests)
# ══════════════════════════════════════════════════
echo -e "${YELLOW}Memory Retrieve Tests (20)${NC}"
test_intent "What's my locker combo again?" "memory_retrieve" 21
test_intent "What medicines am I on?" "memory_retrieve" 22
test_intent "Where did I park the car?" "memory_retrieve" 23
test_intent "What's my blood type?" "memory_retrieve" 24
test_intent "What am I allergic to?" "memory_retrieve" 25
test_intent "Did I finish any books recently?" "memory_retrieve" 26
test_intent "Pull up the flight details I saved" "memory_retrieve" 27
test_intent "Show me the API token I saved" "memory_retrieve" 28
test_intent "When's the vet appointment for Max?" "memory_retrieve" 29
test_intent "What preferences have I set?" "memory_retrieve" 30
test_intent "Do you remember my sister's name?" "memory_retrieve" 31
test_intent "What did I say about sugar and snacks?" "memory_retrieve" 32
test_intent "What's my workout schedule?" "memory_retrieve" 33
test_intent "Show all passwords I've saved" "memory_retrieve" 34
test_intent "Anything due before end of month?" "memory_retrieve" 35
test_intent "What did we talk about earlier?" "memory_retrieve" 36
test_intent "Can you recall my workout schedule?" "memory_retrieve" 37
test_intent "What languages did I say I'm learning?" "memory_retrieve" 38
test_intent "What's my social media limit?" "memory_retrieve" 39
test_intent "Remind me of the grocery list from this morning" "memory_retrieve" 40

# ══════════════════════════════════════════════════
# Web Search Tests (20 tests)
# ══════════════════════════════════════════════════
echo -e "${YELLOW}Web Search Tests (20)${NC}"
test_intent "What's the temperature in Chicago right now?" "web_search" 41
test_intent "Who's the prime minister of UK right now?" "web_search" 42
test_intent "What's the current price of gold per ounce?" "web_search" 43
test_intent "Latest news on GPT-5?" "web_search" 44
test_intent "Eagles score tonight" "web_search" 45
test_intent "When is the next SpaceX Starship launch?" "web_search" 46
test_intent "What's the 10-year Treasury yield right now?" "web_search" 47
test_intent "Top rated headphones under 200 dollars" "web_search" 48
test_intent "When is Diwali this year?" "web_search" 49
test_intent "What happened in the news today?" "web_search" 50
test_intent "Who's the current CEO of OpenAI?" "web_search" 51
test_intent "New Node.js LTS version" "web_search" 52
test_intent "When does the F1 Monaco GP start?" "web_search" 53
test_intent "Best budget gaming PC" "web_search" 54
test_intent "What's the best coffee maker?" "web_search" 55
test_intent "Top rated air purifier" "web_search" 56
test_intent "Who won yesterday's World Series game?" "web_search" 57
test_intent "What's the best smartphone camera?" "web_search" 58
test_intent "Current population of Tokyo" "web_search" 59
test_intent "US CPI print date this month" "web_search" 60

# ══════════════════════════════════════════════════
# General Knowledge Tests (25 tests)
# ══════════════════════════════════════════════════
echo -e "${YELLOW}General Knowledge Tests (25)${NC}"
test_intent "What is the Pythagorean theorem?" "general_knowledge" 61
test_intent "How does RSA encryption work at a high level?" "general_knowledge" 62
test_intent "What is the difference between a stack and a queue?" "general_knowledge" 63
test_intent "Explain how DNS resolution works step-by-step" "general_knowledge" 64
test_intent "Who painted the Mona Lisa?" "general_knowledge" 65
test_intent "What is the difference between RAM and ROM?" "general_knowledge" 66
test_intent "What is a black hole?" "general_knowledge" 67
test_intent "How does compound interest work?" "general_knowledge" 68
test_intent "What are examples of renewable energy?" "general_knowledge" 69
test_intent "Why do leaves change color in the fall?" "general_knowledge" 70
test_intent "What is recursion?" "general_knowledge" 71
test_intent "Explain the concept of supply and demand" "general_knowledge" 72
test_intent "What is the difference between a virus and a bacterium?" "general_knowledge" 73
test_intent "Explain closures in JavaScript" "general_knowledge" 74
test_intent "What are SOLID principles?" "general_knowledge" 75
test_intent "Should I use React or Vue?" "general_knowledge" 76
test_intent "What's the difference between npm and yarn?" "general_knowledge" 77
test_intent "Explain how async/await works" "general_knowledge" 78
test_intent "What are best practices for API design?" "general_knowledge" 79
test_intent "How do I prepare for coding interviews?" "general_knowledge" 80
test_intent "Why is my code not working?" "general_knowledge" 81
test_intent "What does this error mean?" "general_knowledge" 82
test_intent "Can you elaborate?" "general_knowledge" 83
test_intent "Tell me more" "general_knowledge" 84
test_intent "What are your capabilities?" "general_knowledge" 85

# ══════════════════════════════════════════════════
# Greeting Tests (15 tests)
# ══════════════════════════════════════════════════
echo -e "${YELLOW}Greeting Tests (15)${NC}"
test_intent "Heya!" "greeting" 86
test_intent "Morning!" "greeting" 87
test_intent "G'day mate" "greeting" 88
test_intent "Sup" "greeting" 89
test_intent "Hey hey hey!" "greeting" 90
test_intent "Yo, what's up" "greeting" 91
test_intent "Hope you're doing well" "greeting" 92
test_intent "I'm back" "greeting" 93
test_intent "Hi friend" "greeting" 94
test_intent "Yo AI" "greeting" 95
test_intent "Good to see you" "greeting" 96
test_intent "Thanks, that's all for now, bye" "greeting" 97
test_intent "Happy Friday!" "greeting" 98
test_intent "Just checking in" "greeting" 99
test_intent "Hey, long time no see" "greeting" 100

echo ""
echo "=========================================="
echo "Test Results Summary"
echo "=========================================="
echo -e "Total Tests:  $TOTAL_TESTS"
echo -e "${GREEN}Passed:       $PASSED_TESTS${NC}"
echo -e "${RED}Failed:       $FAILED_TESTS${NC}"

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}All tests passed! ✓${NC}"
    exit 0
else
    PASS_RATE=$((PASSED_TESTS * 100 / TOTAL_TESTS))
    echo -e "${YELLOW}Pass Rate:    $PASS_RATE%${NC}"
    exit 1
fi
