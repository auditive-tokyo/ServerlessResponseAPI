import numpy as np
import faiss
from faiss import IndexFlatL2
from typing import Optional, Tuple
from src.utils.dir_config import VECTORS_PATH
from src.utils.logging_config import logger

dimension = 1536
index: Optional[IndexFlatL2] = None
vectors: Optional[np.ndarray] = None

async def load_vectors_and_create_index() -> Tuple[Optional[np.ndarray], Optional[IndexFlatL2]]:
    try:
        vectors = np.load(VECTORS_PATH)
        
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors)
        
        return vectors, index
    except Exception as e:
        logger.error(f"Error loading vectors: {e}")
        return None, None