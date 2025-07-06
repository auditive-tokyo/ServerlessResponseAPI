import json
import os
import aioboto3
import asyncio

from log_config import logger
from create_filter_key import create_filter_keys
from stream_response import generate_stream_response

import importlib.util
if importlib.util.find_spec("dotenv"):
    from dotenv import load_dotenv
    load_dotenv()


MODEL = "gpt-4.1-mini"
VECTOR_SEARCH_FILTER_KEY = os.getenv('VECTOR_SEARCH_FILTER_KEY')

session = aioboto3.Session()

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
        previous_response_id = body.get('previous_response_id', None)

        if not user_message:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'Message is required'})
            }
        
        if VECTOR_SEARCH_FILTER_KEY:
            filters = create_filter_keys(["201", "common"], filter_key=VECTOR_SEARCH_FILTER_KEY)
        else:
            filters = None

        response_generator = generate_stream_response(
            user_message=user_message,
            model=MODEL,
            previous_response_id=previous_response_id,
            filters=filters
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


# テスト実行用
if __name__ == "__main__":    
    # テスト用のイベントデータ
    test_event = {
        'body': json.dumps({
            'message': '部屋は禁煙ですか？',
            'previous_response_id': "resp_6866bef86f9881a2bb7fd2ba330503120dd52a052ded57e1"
        })
    }
    
    # テスト用のコンテキスト
    test_context = {}
    
    # Lambda関数を実行
    result = lambda_handler(test_event, test_context)
    print("Test Result:")
    print(result)