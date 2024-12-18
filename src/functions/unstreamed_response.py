import openai
from openai.error import ServiceUnavailableError
from flask import jsonify
from src.utils.logging_config import logger
import validators
from src.utils.token_utils import count_tokens_with_tiktoken
from src.functions.logdata import log_data

def generate_chat_response(local_model, history, user_id, headers):
    try:
        response = openai.ChatCompletion.create(
            model=local_model,
            messages=list(history[user_id]),
            headers=headers
        )
        return response, None
    except ServiceUnavailableError:
        logger.error("The server is overloaded or not ready yet. Please try again later.")
        return None, jsonify({"error": "The server is overloaded or not ready yet. Please try again later."}), 503
    except Exception as e:
        logger.error(f"Error while generating chat completion: {e}")
        return None, jsonify({"error": "Failed to generate a response."}), 500

def process_chat_response(response, actual_titles, actual_urls, history, user_id, token_limit, local_log_option, user_message, cognito_user_id, matched_ids):
    # AIのレスポンスに参照を追加する
    if actual_titles:
        references = ""
        for title, url in zip(actual_titles, actual_urls):
            if validators.url(url):
                references += f'<br><br><a href="{url}" target="_blank">{title}</a>'
            else:
                references += f'<br><br>{title} ({url})'
        new_message = {"role": "assistant", "content": response['choices'][0]['message']['content'] + references}
    else:
        new_message = {"role": "assistant", "content": response['choices'][0]['message']['content']}

    # トリム後のトークン数を確認
    new_message_tokens = count_tokens_with_tiktoken(new_message["content"])
    logger.info(f"Tokens in final new_message (after AI response): {new_message_tokens}")

    # Check if the new message would cause the total tokens to exceed the limit
    while sum(count_tokens_with_tiktoken(message["content"]) for message in history[user_id]) + count_tokens_with_tiktoken(new_message["content"]) > token_limit:
        # Remove the oldest message
        history[user_id].popleft()

    # Add the new message
    history[user_id].append(new_message)
    
    # get_similar_faiss_id関数でmatched_idがあった場合は初期化
    if 'closest_titles' not in locals():
        closest_titles = []
        combined_scores = []
        closest_vector_indices = []
    
    full_response_content = ""
    log_data(cognito_user_id, local_log_option, user_message, response, full_response_content, actual_urls, closest_titles, combined_scores, closest_vector_indices, matched_ids)

    logger.info(f"Conversation history for user {user_id}: {history[user_id]}")
    return new_message