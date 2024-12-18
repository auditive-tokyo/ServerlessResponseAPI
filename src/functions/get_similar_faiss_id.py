import re
import openai
from src.utils.logging_config import logger

def get_similar_faiss_id(headers, local_model, local_knowledge_about_user, user_message, user_id, history, prefix, combined_list):
    # 既存の履歴から必要なメッセージを抽出
    messages = []

    relevant_messages = [
        msg for msg in history.get(user_id, []) 
        if msg["role"] in ["user", "assistant"] and not msg["content"].startswith(prefix)
    ]

    # 過去の会話履歴が存在する場合のみ、過去の会話履歴を文字列として結合し、メッセージとして追加
    if relevant_messages:
        past_conversation = "\n".join([f"Role: {msg['role']}\nContent: {msg['content']}" for msg in relevant_messages])
        messages.append({"role": "system", "content": f"Here is the past conversation of this user, use it if it helps understanding the context of user's query:\n{past_conversation}\n"})
    
    # ユーザーの背景情報をシステムメッセージとして追加
    if local_knowledge_about_user:
        messages.append({
            "role": "system",
            "content": f"You are the knowledgeable assistant of following entity or association: \n{local_knowledge_about_user}\n"
        })
    
    # ユーザーのクエリに関する指示を追加
    instruction = (
        f"Given the user's query, identify and return the numeric ID(s) from the list of questions and their corresponding IDs: \n{combined_list}.\n\n"
        "If the context of the user's question matches any of the listed questions, return the corresponding ID(s). "
        "Remember to return ONLY the numeric ID(s). If there's no match, return 'None'. You MUST NOT reply to the user's message directly, NOR ASK for further clarification.\n"
    )
    
    messages.extend([
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_message}
    ])
    
    # メッセージの内容を表示
    logger.info("==== Messages List Content ====")
    for msg in messages:
        logger.info(f"Role: {msg['role']}")
        if "questions and their corresponding IDs:" in msg['content']:
            pre_text, post_text = msg['content'].split("questions and their corresponding IDs:")
            logger.info(f"Content: {pre_text}questions and their corresponding IDs:")
            logger.info(post_text)  # combined_listの部分をそのまま表示
        else:
            logger.info(f"Content: {msg['content']}")
    logger.info("===============================")

    # OpenAIのAPIを使用してユーザーメッセージと質問の類似度を計算
    response = openai.ChatCompletion.create(
        model=local_model,
        messages=messages,
        headers=headers,
        temperature=0.0
    )
    logger.info(f"OpenAI API Response: {response.choices[0].message['content']}")
    
    # レスポンスからIDを抽出
    matched_ids = re.findall(r'\d+', response.choices[0].message['content'])
    logger.info(f"Matched IDs: {matched_ids}")
    
    return matched_ids[:4]  # 最初の4つのIDを返す