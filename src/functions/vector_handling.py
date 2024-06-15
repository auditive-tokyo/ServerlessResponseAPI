import os
import numpy as np
import faiss
from faiss import IndexFlatL2
from typing import Optional, Any, Tuple
from src.utils.file_utils import generate_folder_name

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

dimension = 1536
index: Optional[IndexFlatL2] = None
vectors: Optional[np.ndarray] = None

def load_vectors_and_create_index(cognito_user_id: Any) -> Tuple[str, Any, IndexFlatL2]:
    folder_name = generate_folder_name(cognito_user_id)
    vectors_path = os.path.join(ROOT_DIR, folder_name, 'vectors.npy')
    vectors = np.load(vectors_path)
    
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    
    return vectors_path, vectors, index