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

## Deploy
Run `sam deploy` (use --guided for the first time)

## GitHub Actions Integration

You can integrate this repository from another GitHub Actions workflow using the `actions/checkout` step with the `repository` option:

```yaml
- name: Checkout ServerlessEmbedAI
  uses: actions/checkout@v4
  with:
    repository: auditive-tokyo/serverlessembeddai
    ref: main
```