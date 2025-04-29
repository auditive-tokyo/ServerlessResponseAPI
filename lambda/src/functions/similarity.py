import numpy as np
from typing import List, Optional
from faiss import IndexFlatL2
from src.utils.logging_config import logger

async def similarity(
    embedded_message: Optional[np.ndarray],
    index: IndexFlatL2,
    similarity_threshold: float,
    top_k: int = 10
) -> List[int]:
    try:
        if embedded_message is None:
            return []
            
        # 入力ベクトルを2D配列に変換
        query_vector = np.array([embedded_message], dtype=np.float32)
        
        # FAISSで類似度検索
        distances, indices = index.search(query_vector, top_k)
        
        # しきい値以上の類似度を持つインデックスを抽出
        similar_indices = [
            idx for idx, dist in zip(indices[0], distances[0])
            if 1 / (1 + dist) >= similarity_threshold
        ]
        
        return similar_indices[:4]  # 上位4件まで返す
        
    except Exception as e:
        logger.error(f"Error in similarity search: {e}")
        return []