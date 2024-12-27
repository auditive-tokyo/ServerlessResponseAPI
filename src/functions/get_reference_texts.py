from typing import List, Dict
import json
from src.utils.dir_config import REFERENCE_PATH
from src.utils.logging_config import logger

async def get_reference_texts(similar_indices: List[int]) -> List[Dict[str, str]]:
    try:
        # reference.jsonを読み込む
        with open(REFERENCE_PATH, 'r', encoding='utf-8') as f:
            references = json.load(f)
        
        # FAISSのインデックスに対応するリファレンスを取得
        reference_texts = []
        for idx in similar_indices:
            if 0 <= idx < len(references):
                ref = references[idx]
                reference_texts.append({
                    "title": ref["title"],
                    "url": ref["url"],
                    "text": ref["text"]
                })
                logger.debug(f"Found reference: {ref['title']}")
            else:
                logger.warning(f"Index {idx} out of range")
        
        return reference_texts

    except Exception as e:
        logger.error(f"Error getting reference texts: {e}")
        return []