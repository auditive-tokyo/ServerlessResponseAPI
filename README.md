# Chatbot with Serverless Architecture

A serverless-architected chatbot that utilizes embeddings and FAISS for similarity search indexing, featuring real-time streaming responses via Server-Sent Events (SSE).

## Features
- Serverless architecture using AWS Lambda
- Real-time streaming responses using Server-Sent Events (SSE)
- Similarity search using FAISS indexing
- Streaming responses from OpenAI's API
- Chat history management with DynamoDB
- Local development environment with SAM CLI

## Local Development
1. `sam build`
2. `sam local invoke ChatFunction --event events/event.json` for testing

## Deploy
`sam deploy --guided` for 1st time