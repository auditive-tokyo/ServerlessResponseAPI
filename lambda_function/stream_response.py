import json
from typing import List, Dict, AsyncGenerator, Union
from openai import AsyncOpenAI
from log_config import logger
import os

import importlib.util
if importlib.util.find_spec("dotenv"):
    from dotenv import load_dotenv
    load_dotenv()


PROMPT_TEMPLATE = os.getenv('PROMPT_TEMPLATE')
OPENAI_VECTOR_STORE_ID = os.getenv('OPENAI_VECTOR_STORE_ID')

async def generate_stream_response(
    user_message: str,
    model: str,
    previous_response_id: str = None,
    filters: List[Dict[str, Union[str, int, bool]]] = None
) -> AsyncGenerator[str, None]:
    try:
        client = AsyncOpenAI()

        logger.info(f"Response API File Search開始: '{user_message}' (Vector Store: {OPENAI_VECTOR_STORE_ID})")

        # File Searchツールの有無を分岐
        tools = []
        tool_choice = None

        if OPENAI_VECTOR_STORE_ID:
            tools_config = {
                "type": "file_search",
                "vector_store_ids": [OPENAI_VECTOR_STORE_ID],
                "max_num_results": 10,
                "ranking_options": {"score_threshold": 0.2}
            }
            if filters:
                tools_config["filters"] = {
                    "type": "or",
                    "filters": filters
                }
            tools = [tools_config]
            tool_choice = "auto"
        else:
            logger.warning("No Vector Store ID provided, skipping file search tool")

        request_payload = {
            "model": model,
            "prompt": {
                "id": PROMPT_TEMPLATE
            },
            "input": [{"role": "user", "content": user_message}],
            "tools": tools,
            "temperature": 0.7,
            "truncation": "auto",
            "stream": True
        }
        if tool_choice:
            request_payload["tool_choice"] = tool_choice

        if previous_response_id:
            request_payload["previous_response_id"] = previous_response_id
            logger.info(f"Previous Response ID: {previous_response_id}")

        response_stream = await client.responses.create(**request_payload)
        logger.debug(f"Response Stream initiated")

        full_response_content = ""
        current_response_id = None

        async for chunk in response_stream:
            logger.debug(f"Received chunk: {chunk}")
            if hasattr(chunk, 'type') and chunk.type == 'response.created':
                if hasattr(chunk, 'response') and hasattr(chunk.response, 'id') and not current_response_id:
                    current_response_id = chunk.response.id
                    logger.info(f"Response ID: {current_response_id}")
                    yield f"data: {json.dumps({'response_id': current_response_id})}\n\n"

            if hasattr(chunk, 'type') and chunk.type == 'response.output_text.delta':
                if hasattr(chunk, 'delta') and chunk.delta:
                    content = chunk.delta
                    full_response_content += content
                    yield f"data: {json.dumps({'content': content})}\n\n"

        final_data = {
            'response_id': current_response_id,
            'completed': True
        }
        yield f"data: {json.dumps(final_data)}\n\n"

        logger.info("Response API stream completed successfully")

    except Exception as e:
        logger.error(f"Error in generate_stream_response: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        if 'client' in locals():
            await client.close()
