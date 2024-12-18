import json
import openai
import validators
from src.functions.logdata import log_data
from src.utils.token_utils import count_tokens_with_tiktoken
from src.utils.logging_config import logger
from openai.error import ServiceUnavailableError

def generate(user_id, user_message, user_history, token_limit, local_log_option, cognito_user_id, local_model, headers, actual_titles, actual_urls, closest_titles, combined_scores, closest_vector_indices, matched_ids, history):
    logger.info("generate function has started")
    try:
        logger.info("Attempting to create a streaming response...")
        response_stream = openai.ChatCompletion.create(
            model=local_model,
            messages=list(user_history),
            headers=headers,
            stream=True
        )
        logger.info("Streaming response created successfully.")
        
        full_response_content = ""  # AIからの完全なレスポンスを保存するための変数

        for chunk in response_stream:
            if 'content' in chunk['choices'][0]['delta']:
                content = chunk['choices'][0]['delta']['content']
                full_response_content += content  # レスポンスを連結
                
                yield f"data: {json.dumps({'content': content})}\n\n"

        # AIのレスポンスに参照を追加する
        if actual_titles:
            references = ""
            for title, url in zip(actual_titles, actual_urls):
                if validators.url(url):
                    references += f'<br><br><a href="{url}" target="_blank">{title}</a>'
                else:
                    references += f'<br><br>{title} ({url})'
            new_message = {"role": "assistant", "content": full_response_content + references}
            yield f"data: {json.dumps({'content': references})}\n\n"
        else:
            new_message = {"role": "assistant", "content": full_response_content}

        # トリム後のトークン数を確認
        new_message_tokens = count_tokens_with_tiktoken(new_message["content"])
        logger.info(f"Tokens in final new_message (after AI response): {new_message_tokens}")

        # Check if the new message would cause the total tokens to exceed the limit
        while sum(count_tokens_with_tiktoken(message["content"]) for message in user_history if isinstance(message, dict) and "content" in message) + new_message_tokens > token_limit:
            # Remove the oldest message from user_history
            user_history.popleft()

        # Add the new message to user_history
        user_history.append(new_message)
        
        response = ""
        log_data(cognito_user_id, local_log_option, user_message, response, full_response_content, actual_urls, closest_titles, combined_scores, closest_vector_indices, matched_ids)
        
        # Update the global history with the user's updated history
        history[user_id] = user_history

        logger.info(f"Conversation history for user {user_id}: {history[user_id]}")

    except ServiceUnavailableError:
        yield f"data: {json.dumps({'error': 'The server is overloaded or not ready yet. Please try again later.'})}\n\n"
    except Exception as e:
        logger.error(f"Error while generating chat completion: {e}")
        yield f"data: {json.dumps({'error': 'Failed to generate a response.'})}\n\n"