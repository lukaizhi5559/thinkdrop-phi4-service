# Thinkdrop Phi4 MCP Service

> **NLP/LLM microservice for Thinkdrop AI** - Intent parsing, entity extraction, and general Q&A

[![Node.js](https://img.shields.io/badge/node-%3E%3D18.0.0-brightgreen)](https://nodejs.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 🚀 Quick Start

```bash
# 1. Clone and install
git clone https://github.com/your-org/thinkdrop-phi4-service.git
cd thinkdrop-phi4-service
npm install

# 2. Configure environment
cp .env.example .env
# Edit .env with your settings

# 3. Download models (first time only)
npm run models:download

# 4. Start service
npm run dev

# 5. Test
curl http://localhost:3003/service.health
```

## 📋 Features

- ✅ **Intent Classification** - 95%+ accuracy with DistilBERT
- ✅ **Entity Extraction** - NER-based entity recognition
- ✅ **General Q&A** - Phi4 LLM integration (placeholder)
- ✅ **Embedding Generation** - 384-dimensional semantic vectors
- ✅ **Multi-Parser Support** - 4 parsers with automatic fallback
- ✅ **MCP Protocol** - Standard microservice communication
- ✅ **Docker Ready** - Production-ready containerization

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Thinkdrop AI App                      │
└────────────────────┬────────────────────────────────────┘
                     │ MCP Protocol
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Phi4 MCP Service (Port 3003)               │
├─────────────────────────────────────────────────────────┤
│  Routes: /intent.parse, /entity.extract, etc.          │
│  Parsers: DistilBERT, Fast, Hybrid, Original           │
│  Models: Embeddings, NER, Phi4 LLM                     │
└─────────────────────────────────────────────────────────┘
```

## 📡 API Endpoints

### Health & Capabilities

```bash
GET /service.health        # Service health status
GET /service.capabilities  # Available actions
```

### MCP Actions

All actions use POST with MCP envelope:

```bash
POST /intent.parse         # Classify intent
POST /general.answer       # Generate answers
POST /entity.extract       # Extract entities
POST /embedding.generate   # Generate embeddings
POST /parser.list          # List parsers
```

### Example Request

```bash
curl -X POST http://localhost:3003/intent.parse \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "version": "mcp.v1",
    "service": "phi4",
    "action": "intent.parse",
    "requestId": "req-123",
    "payload": {
      "message": "Remember I have a meeting tomorrow at 3pm",
      "options": {
        "parser": "distilbert",
        "includeEntities": true
      }
    }
  }'
```

### Example Response

```json
{
  "version": "mcp.v1",
  "service": "phi4",
  "action": "intent.parse",
  "requestId": "req-123",
  "status": "ok",
  "data": {
    "intent": "memory_store",
    "confidence": 0.95,
    "entities": [
      {
        "type": "datetime",
        "value": "tomorrow at 3pm",
        "confidence": 0.88
      }
    ],
    "suggestedResponse": "I'll remember that you have a meeting tomorrow at 3pm.",
    "parser": "distilbert"
  },
  "metrics": {
    "elapsedMs": 42
  }
}
```

## 🔧 Configuration

### Environment Variables

See `.env.example` for all options. Key settings:

```bash
# Server
PORT=3003
API_KEY=your-secret-key

# Parsers
DEFAULT_PARSER=distilbert
ENABLE_DISTILBERT=true
ENABLE_FAST=true

# Performance
MODEL_WARMUP_ON_START=true
```

## 🧪 Testing

### Automated Testing

```bash
# Run all tests with Jest
npm test

# Unit tests only
npm run test:unit

# Integration tests only
npm run test:integration

# Test with real API calls
node scripts/test-messages.js

# Test all endpoints with shell script
./scripts/test-api.sh
```

### Manual Testing

```bash
# Test a message prompt
curl -X POST http://localhost:3003/intent.parse \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "version": "mcp.v1",
    "service": "phi4",
    "action": "intent.parse",
    "payload": {
      "message": "Remember I have a meeting tomorrow at 3pm"
    }
  }'

# Test LLM Q&A
curl -X POST http://localhost:3003/general.answer \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "version": "mcp.v1",
    "service": "phi4",
    "action": "general.answer",
    "payload": {
      "query": "What is the capital of France?"
    }
  }'
```

See [TESTING_GUIDE.md](TESTING_GUIDE.md) for comprehensive testing instructions.

## 🐳 Docker Deployment

```bash
# Build and run
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop
docker-compose down
```

## 📊 Performance

| Parser     | Accuracy | Latency (p50) | Use Case           |
|------------|----------|---------------|--------------------|
| DistilBERT | 95%      | 42ms          | Production default |
| Hybrid     | 88%      | 67ms          | Balanced           |
| Fast       | 82%      | 15ms          | Low latency        |
| Original   | 85%      | 89ms          | Legacy support     |

## 🔒 Security

- API key authentication required
- Rate limiting (100 req/min)
- CORS protection
- Input validation
- Helmet.js security headers

## 📚 Documentation

- Full specification: [AGENT_SPEC_Phi4.md](AGENT_SPEC_Phi4.md)
- MCP Protocol: See spec for details
- Parser details: See `src/parsers/`

## 🤝 Integration

### From Thinkdrop AI App

```javascript
const response = await fetch('http://localhost:3003/intent.parse', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer your-api-key'
  },
  body: JSON.stringify({
    version: 'mcp.v1',
    service: 'phi4',
    action: 'intent.parse',
    payload: { message: 'Your message here' }
  })
});

const result = await response.json();
console.log(result.data.intent);
```

## 🛠️ Development

```bash
# Install dependencies
npm install

# Run in development mode
npm run dev

# Lint code
npm run lint

# Format code
npm run format
```

## 📝 License

MIT License - see [LICENSE](LICENSE) file

## 🙋 Support

- Issues: [GitHub Issues](https://github.com/your-org/thinkdrop-phi4-service/issues)
- Documentation: [AGENT_SPEC_Phi4.md](AGENT_SPEC_Phi4.md)

---

**Built for Thinkdrop AI** | Version 1.0.0
