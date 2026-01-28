# Serverless Chatbot with OpenAI Vector Store

A serverless chatbot that leverages OpenAI's Response API and Vector Store for similarity search and real-time streaming responses.

## Features

- Serverless architecture using AWS Lambda
- Real-time streaming responses (SSE)
- Similarity search using OpenAI Vector Store (embedding-based retrieval)
- Simple integration with OpenAI's Response API
- Easy deployment with AWS SAM CLI

## How it works

The chatbot uses OpenAI's Vector Store to find and reference similar documents based on user queries via embeddings, and streams responses in real time.

## Local Development

1. `npm run build` (if using TypeScript)
2. `sam build`
3. `sam local start-api` for testing

## Testing

Tests are **integration tests** that call the actual OpenAI API. You need valid API credentials to run them.

### Setup

```bash
npm install
```

### Environment Variables

Create a `.env` file for testing:

```
OPENAI_API_KEY=your-api-key           # Required
OPENAI_VECTOR_STORE_ID=vs_xxxxx       # Required for file search tests
PROMPT_TEMPLATE_ID=pmpt_xxxxx         # Optional
```

### Run Tests

```bash
# Run all tests
npm test

# Run specific test file
npm test stream_response
npm test chat_handler

# Watch mode
npm run test:watch

# With coverage
npm run test:coverage
```

### Test Structure

```
test/
├── chat_handler.test.ts    # Request validation tests
└── stream_response.test.ts # OpenAI API integration tests
```

## Deploy

Run `sam deploy` (use --guided for the first time)

## CI/CD

### SonarCloud Integration

On push to `main`, GitHub Actions runs tests with coverage and sends results to SonarCloud.

**Required GitHub Secrets:**
| Secret | Description |
|--------|-------------|
| `SONAR_TOKEN` | SonarCloud token (for CI IT Test) |
| `OPENAI_API_KEY` | For running integration tests |
| `VECTOR_STORE_ID` | For file search tests |
| `PROMPT_TEMPLATE_ID` | Optional |

### GitHub Actions Integration

You can integrate this repository from another GitHub Actions workflow using the `actions/checkout` step with the `repository` option:

```yaml
- name: Checkout ServerlessEmbedAI
  uses: actions/checkout@v4
  with:
    repository: auditive-tokyo/serverlessembeddai
    ref: main
```
