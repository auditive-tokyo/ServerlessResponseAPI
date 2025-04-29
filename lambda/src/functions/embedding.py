from openai import AsyncOpenAI
import numpy as np
from typing import Dict, Any, Optional

async def embedding_user_message(message: str, headers: Dict[str, Any]) -> Optional[np.ndarray]:
    try:
        # AsyncOpenAIクライアントの初期化
        client = AsyncOpenAI(api_key=headers["Authorization"].split(" ")[1])
        
        # 埋め込みの生成
        response = await client.embeddings.create(
            model="text-embedding-3-large",
            input=message
        )
        
        if response and hasattr(response, 'data') and response.data:
            embedding = response.data[0].embedding
            return np.array(embedding, dtype=np.float32)
        
        print("No embedding data in response")
        return None
        
    except Exception as e:
        print(f"Error in embedding: {e}")
        return None