import openai
from src.utils.logging_config import logger

def embedding_user_message(user_message: str, headers: dict):
    try:
        embedding_result = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=user_message,
            headers=headers
        )
        user_message_vector = embedding_result['data'][0]['embedding']
        return user_message_vector
    except Exception as e:
        logger.error(f"Error while embedding user message: {e}")
        return None