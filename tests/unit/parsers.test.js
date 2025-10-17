/**
 * Unit tests for parsers
 */

const FastIntentParser = require('../../src/parsers/FastIntentParser.cjs');

describe('FastIntentParser', () => {
  let parser;

  beforeAll(async () => {
    parser = new FastIntentParser();
    await parser.initialize();
  });

  test('should classify memory_store intent', async () => {
    const result = await parser.parse('Remember I have a meeting tomorrow');
    expect(result.intent).toBe('memory_store');
    expect(result.confidence).toBeGreaterThan(0.5);
  });

  test('should classify memory_retrieve intent', async () => {
    const result = await parser.parse('What meetings do I have tomorrow?');
    expect(result.intent).toBe('memory_retrieve');
    expect(result.confidence).toBeGreaterThan(0.5);
  });

  test('should classify command intent', async () => {
    const result = await parser.parse('Take a screenshot');
    expect(result.intent).toBe('command');
    expect(result.confidence).toBeGreaterThan(0.5);
  });

  test('should classify question intent', async () => {
    const result = await parser.parse('What is the capital of France?');
    expect(result.intent).toBe('question');
    expect(result.confidence).toBeGreaterThan(0.5);
  });

  test('should classify greeting intent', async () => {
    const result = await parser.parse('Hello there');
    expect(result.intent).toBe('greeting');
    expect(result.confidence).toBeGreaterThan(0.5);
  });

  test('should handle empty message', async () => {
    const result = await parser.parse('');
    expect(result).toHaveProperty('intent');
    expect(result).toHaveProperty('confidence');
  });
});
