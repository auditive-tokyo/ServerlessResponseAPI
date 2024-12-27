import json
from typing import List, Dict, AsyncGenerator, Union
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam
from src.utils.logging_config import logger

async def generate_stream_response(
    user_message: str,
    history: List[Dict[str, str]],
    reference_texts: List[Dict[str, str]],
    model: str,
    headers: Dict[str, str],
    browser_id: str,
    save_history
) -> AsyncGenerator[str, None]:
    try:
        # AsyncOpenAIクライアントの初期化
        client = AsyncOpenAI()

        # メッセージリストの型を明示的に定義
        formatted_messages: List[Union[
            ChatCompletionSystemMessageParam,
            ChatCompletionUserMessageParam,
            ChatCompletionAssistantMessageParam
        ]] = []
        for msg in history:
            if msg["role"] == "system":
                formatted_messages.append(ChatCompletionSystemMessageParam(role="system", content=msg["content"]))
            elif msg["role"] == "user":
                formatted_messages.append(ChatCompletionUserMessageParam(role="user", content=msg["content"]))
            elif msg["role"] == "assistant":
                formatted_messages.append(ChatCompletionAssistantMessageParam(role="assistant", content=msg["content"]))

        # ユーザーメッセージを追加
        formatted_messages.append(ChatCompletionUserMessageParam(role="user", content=user_message))
        
        # OpenAIのストリームレスポンス生成
        response_stream = await client.chat.completions.create(
            model=model,
            messages=formatted_messages,
            stream=True
        )
        logger.debug(f"Response Stream: {response_stream}")

        full_response_content = ""  # AIからの完全なレスポンスを保存するための変数

        async for chunk in response_stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content)
                full_response_content += content
                
                yield f"data: {json.dumps({'content': content})}\n\n"

        # 参照URLのリストを作成
        reference_urls = [ref['url'] for ref in reference_texts]
        # レスポンスの最後に参照情報を追加
        references_text = "\n\nReferences:\n" + "\n".join(reference_urls)
        yield f"data: {json.dumps({'content': references_text})}\n\n"
        
        # 履歴に追加するための完全なレスポンス（参照情報付き）
        history.append({
            "role": "assistant", 
            "content": full_response_content + references_text
        })

        # 履歴を保存
        response = await save_history(browser_id, history)
        logger.info(response)
        
    except Exception as e:
        logger.error(f"Error in generate_stream_response: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"