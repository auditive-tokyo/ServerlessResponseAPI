import json
import os
import aioboto3
import asyncio

from src.utils.logging_config import logger
from src.functions.embedding import embedding_user_message
from src.functions.vector_handling import load_vectors_and_create_index
# from src.functions.get_similar_faiss_id import get_similar_faiss_id
from src.functions.similarity import similarity
from src.functions.get_reference_texts import get_reference_texts
from src.functions.dynamo_functions import get_chat_history, save_chat_history
from src.functions.stream_response import generate_stream_response

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MODEL = "gpt-4o-mini"
SIMILARITY_THRESHOLD = 0.67

session = aioboto3.Session()
TABLE_NAME = "test_chat_history"

async def lambda_handler(event, context):
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

        # 履歴の取得
        history = await get_chat_history(browser_id)
        logger.debug(f"history exists: {len(history)}")

        # OpenAI APIのヘッダーを設定
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        # メッセージのembedding
        embedded_message = await embedding_user_message(user_message, headers)
        # print(f"embeded_message: {embedded_message[:16]}")
        
        # ベクトルとインデックスの読み込み
        vectors, index = await load_vectors_and_create_index()
        # print(f"vector, index: {vectors}\n{index}")
        if vectors is None or index is None:
            return {
                'statusCode': 500,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'Failed to load vectors'})
            }

        # 類似度の計算
        similar_indices = await similarity(embedded_message, index, SIMILARITY_THRESHOLD)
        logger.debug(f"Similar Indices: {similar_indices}")

        # 類似テキストの取得
        reference_texts = await get_reference_texts(similar_indices)
        logger.debug(f"Number of reference texts found: {len(reference_texts)}")
        
        # 新しい参照テキストを追加
        for ref in reference_texts:
            history.append({
                "role": "system",
                "content": f"Reference: {ref['title']}\n{ref['text']}"
            })

        # チャットレスポンスの生成（参照テキストも渡す）
        response_generator = generate_stream_response(
            user_message=user_message,
            history=history,
            reference_texts=reference_texts,
            model=MODEL,
            headers=headers,
            browser_id=browser_id,
            save_history=save_chat_history
        )

        # ジェネレーターを反復処理して応答を生成
        responses = []
        async for chunk in response_generator:
            responses.append(chunk)

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'text/event-stream',
                'Connection': 'keep-alive'
            },
            'body': ''.join(responses)
        }

    except Exception as e:
        logger.error(f"Error in handler: {e}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': str(e)})
        }


# Lambda用のエントリーポイントラッパー
def handler(event, context):
    return asyncio.get_event_loop().run_until_complete(lambda_handler(event, context))