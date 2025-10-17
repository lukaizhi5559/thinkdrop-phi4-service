/**
 * Integration tests for API endpoints
 */

const request = require('supertest');
const app = require('../../src/server');

describe('API Integration Tests', () => {
  describe('GET /service.health', () => {
    test('should return health status', async () => {
      const response = await request(app)
        .get('/service.health')
        .expect(200);

      expect(response.body).toHaveProperty('service', 'phi4');
      expect(response.body).toHaveProperty('status');
      expect(response.body).toHaveProperty('version');
    });
  });

  describe('GET /service.capabilities', () => {
    test('should return capabilities', async () => {
      const response = await request(app)
        .get('/service.capabilities')
        .expect(200);

      expect(response.body).toHaveProperty('capabilities');
      expect(response.body.capabilities).toHaveProperty('actions');
      expect(response.body.capabilities.actions).toBeInstanceOf(Array);
    });
  });

  describe('POST /intent.parse', () => {
    test('should parse intent successfully', async () => {
      const response = await request(app)
        .post('/intent.parse')
        .send({
          version: 'mcp.v1',
          service: 'phi4',
          action: 'intent.parse',
          requestId: 'test-123',
          payload: {
            message: 'Remember I have a meeting tomorrow',
            options: {
              parser: 'fast'
            }
          }
        })
        .expect(200);

      expect(response.body.status).toBe('ok');
      expect(response.body.data).toHaveProperty('intent');
      expect(response.body.data).toHaveProperty('confidence');
    });

    test('should reject invalid request', async () => {
      const response = await request(app)
        .post('/intent.parse')
        .send({
          version: 'mcp.v1',
          service: 'phi4',
          action: 'intent.parse',
          payload: {}
        })
        .expect(400);

      expect(response.body.status).toBe('error');
    });
  });
});
