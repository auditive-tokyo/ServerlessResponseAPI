from typing import List, Dict
import aioboto3
from datetime import datetime, timedelta
from src.utils.logging_config import logger

session = aioboto3.Session()
TABLE_NAME = "test_chat_history"

async def get_chat_history(browser_id: str) -> List[Dict[str, str]]:
    try:
        async with session.resource('dynamodb') as dynamodb:
            table = await dynamodb.Table(TABLE_NAME)
            
            response = await table.get_item(
                Key={'browser_id': browser_id},
                ConsistentRead=True
            )
            
            return response.get('Item', {}).get('history', [])
            
    except Exception as e:
        logger.error(f"DynamoDB get_item error: {str(e)}")
        raise e


async def save_chat_history(
    browser_id: str,
    history: List[Dict[str, str]]
) -> Dict:
    try:
        async with session.resource('dynamodb') as dynamodb:
            table = await dynamodb.Table(TABLE_NAME)
            expiration_time = int((datetime.now() + timedelta(hours=1)).timestamp())
            
            response = await table.put_item(
                Item={
                    'browser_id': browser_id,
                    'history': history,
                    'ttl': expiration_time
                }
            )

            return response
            
    except Exception as e:
        logger.error(f"DynamoDB put_item error: {str(e)}")
        raise e