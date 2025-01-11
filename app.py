import json
import os
import aioboto3
import asyncio

from src.utils.logging_config import logger
from src.functions.embedding import embedding_user_message
from src.functions.vector_handling import load_vectors_and_create_index
from src.functions.similarity import similarity
from src.functions.get_reference_texts import get_reference_texts
from src.functions.dynamo_functions import get_chat_history, save_chat_history
from src.functions.stream_response import generate_stream_response

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MODEL = "gpt-4o-mini"
SIMILARITY_THRESHOLD = 0.67

session = aioboto3.Session()
TABLE_NAME = "test_chat_history"

def lambda_handler(event, context):
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_async_handler(event, context))
    except Exception as e:
        logger.error(f"Error in lambda_handler: {e}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': str(e)})
        }

async def _async_handler(event, context):
    try:        
        if not event.get('body'):
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'Request body is required'})
            }

        body = json.loads(event.get('body', '{}'))
        user_message = body.get('message', '')
        browser_id = body.get('browser_id', '')

        if not user_message:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'Message is required'})
            }

        if not browser_id:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'You need to enable local storage or cookie'})
            }

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

        history, embedded_message, (vectors, index) = await asyncio.gather(
            get_chat_history(browser_id),
            embedding_user_message(user_message, headers),
            load_vectors_and_create_index()
        )
        
        if vectors is None or index is None:
            return {
                'statusCode': 500,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'Failed to load vectors'})
            }

        similar_indices = await similarity(embedded_message, index, SIMILARITY_THRESHOLD)
        reference_texts = await get_reference_texts(similar_indices)
        
        for ref in reference_texts:
            history.append({
                "role": "system",
                "content": f"Reference: {ref['title']}\n{ref['text']}"
            })

        response_generator = generate_stream_response(
            user_message=user_message,
            history=history,
            reference_texts=reference_texts,
            model=MODEL,
            headers=headers,
            browser_id=browser_id,
            save_history=save_chat_history
        )

        responses = []
        async for chunk in response_generator:
            responses.append(chunk)

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Transfer-Encoding': 'chunked'
            },
            'body': json.dumps({
                'isBase64Encoded': False,
                'chunks': responses
            })
        }

    except Exception as e:
        logger.error(f"Error in handler: {e}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': str(e)})
        } 