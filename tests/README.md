# Intent Classification Test Suite

This directory contains automated tests for the DistilBERT intent parser to ensure accurate classification across all intent types.

## Quick Start

```bash
# Make sure the phi4 service is running
npm run dev

# In another terminal, run the tests
npm test

# Run only intent classification tests
npm test -- intent-classification.test.js

# Run with verbose output
npm test -- intent-classification.test.js --verbose
```

## Test Structure

### Test Cases (`tests/unit/intent-classification.test.js`)

The test suite includes **70+ test cases** covering:

- **Web Search** (15 tests): Factual queries, product recommendations, real-time data
- **General Knowledge** (7 tests): Definitions, explanations, static facts
- **Command Automate** (6 tests): System commands, app launches, automation
- **Screen Intelligence** (6 tests): Screen content analysis, OCR, visible elements
- **Memory Store** (4 tests): Storing information for later recall
- **Memory Retrieve** (3 tests): Recalling stored information
- **Greeting** (4 tests): Conversational greetings
- **Edge Cases** (3 tests): Ambiguous or tricky queries

### Confidence Thresholds

Each intent has a minimum confidence threshold:

| Intent | Threshold | Reason |
|--------|-----------|--------|
| `command_automate` | 0.70 | Security implications - must be certain |
| `memory_store` | 0.65 | Important to distinguish from general questions |
| `memory_retrieve` | 0.65 | Important to distinguish from general questions |
| `screen_intelligence` | 0.65 | Requires screen context |
| `web_search` | 0.60 | Default for factual queries |
| `general_knowledge` | 0.60 | Static information |
| `greeting` | 0.60 | Conversational |
| `question` | 0.50 | Often combined with other intents |

## Adding New Test Cases

When you discover a misclassification:

1. **Add it to the test suite** in `intent-classification.test.js`:

```javascript
{
  query: "Your misclassified query here",
  expected: "correct_intent",
  minConfidence: 0.65, // Optional, uses intent default
  description: "Brief description of what's being tested"
}
```

2. **Add training examples** to the DistilBERT parser:
   - File: `src/parsers/DistilBertIntentParser.cjs`
   - Find the correct intent section
   - Add similar examples to strengthen the pattern

3. **Restart the phi4 service** to reload the parser:
```bash
# The service will auto-restart if using nodemon
# Or manually restart:
npm run dev
```

4. **Run the tests** to verify the fix:
```bash
npm test -- intent-classification.test.js
```

## Test Output

### Successful Run
```
PASS  tests/unit/intent-classification.test.js
  Intent Classification Test Suite
    Intent: WEB_SEARCH
      ✓ should classify "Who's the best jumper in the world" as web_search (85ms)
      ✓ should classify "What's the best winter jacket" as web_search (72ms)
      ...
    Intent: COMMAND_AUTOMATE
      ✓ should classify "Open my email" as command_automate (68ms)
      ...

📊 Classification Report:
   Total tests: 70
   Correct: 68
   Accuracy: 97.14%
   Avg confidence (correct): 0.847
```

### Failed Test
```
FAIL  tests/unit/intent-classification.test.js
  Intent Classification Test Suite
    Intent: WEB_SEARCH
      ✕ should classify "What's the best winter jacket" as web_search (82ms)

❌ Misclassifications:
   "What's the best winter jacket"
      Expected: web_search, Got: command_automate (confidence: 0.523)
```

## Confidence Threshold Fallback

The orchestrator (`src/main/services/mcp/nodes/parseIntent.cjs`) implements smart fallback logic:

- **`command_automate` < 0.7** → Falls back to `web_search`
  - Reason: Security implications - better to search than execute wrong command
  
- **`screen_intelligence` < 0.5** → Falls back to `general_knowledge`
  - Reason: Very low confidence likely means it's a general question
  
- **Any intent < 0.4** → Falls back to `web_search`
  - Reason: Extremely low confidence - safest to search

These thresholds prevent misclassifications from causing incorrect behavior.

## Continuous Improvement

### Weekly Review Process

1. **Check test results** from CI/CD
2. **Review misclassification logs** from production
3. **Add new test cases** for common failures
4. **Update training examples** in DistilBERT parser
5. **Re-run tests** to verify improvements

### Monitoring Metrics

Track these metrics over time:
- Overall accuracy (target: >90%)
- Per-intent accuracy
- Average confidence scores
- Fallback frequency

### When to Retrain

Consider fine-tuning the DistilBERT model if:
- Accuracy drops below 85%
- Specific intent consistently fails (< 70% accuracy)
- Adding training examples doesn't improve results
- New intent types are needed

## CI/CD Integration

Add to your CI pipeline (`.github/workflows/test.yml`):

```yaml
name: Intent Classification Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-node@v2
        with:
          node-version: '18'
      
      - name: Install dependencies
        run: npm install
        working-directory: mcp-services/thinkdrop-phi4-service
      
      - name: Start phi4 service
        run: npm run dev &
        working-directory: mcp-services/thinkdrop-phi4-service
      
      - name: Wait for service
        run: sleep 30
      
      - name: Run intent tests
        run: npm test -- intent-classification.test.js
        working-directory: mcp-services/thinkdrop-phi4-service
```

## Troubleshooting

### Service Not Running
```
Error: Phi4 service is not running at http://127.0.0.1:3003
```
**Solution**: Start the service with `npm run dev` before running tests.

### Low Accuracy
```
Accuracy: 72.14%
```
**Solution**: 
1. Review misclassified queries in test output
2. Add similar examples to DistilBERT training data
3. Restart service and re-run tests

### Timeout Errors
```
Timeout - Async callback was not invoked within the 10000 ms timeout
```
**Solution**: Increase timeout in test file or check if service is overloaded.

## Future Enhancements

- [ ] Add user feedback collection UI
- [ ] Implement active learning pipeline
- [ ] Create confusion matrix visualization
- [ ] Add A/B testing framework
- [ ] Build synthetic data generation
- [ ] Add performance benchmarks
- [ ] Create intent classification dashboard

## Resources

- [DistilBERT Documentation](https://huggingface.co/docs/transformers/model_doc/distilbert)
- [Jest Testing Framework](https://jestjs.io/)
- [Intent Classification Best Practices](https://www.rasa.com/docs/rasa/nlu-training-data/)
