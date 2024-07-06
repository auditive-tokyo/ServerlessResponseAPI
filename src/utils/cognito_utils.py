from datetime import datetime
from src.schema.logging_config import logger
from src.utils.cache_utils import cache_lock, cognito_cache
from src.functions.vector_handling import load_vectors_and_create_index
from src.utils.file_utils import get_file_path
from src.schema.config_manager import load_settings

def set_cognito_data(cognito_user_id):
    try:
        # 設定をロード
        settings = load_settings(cognito_user_id)
        
        # reference.jsonのpath取得
        file_path, data, documents = get_file_path(cognito_user_id)
        
        # vectors_pathの取得とインデックスの作成
        vectors_path, vectors, index = load_vectors_and_create_index(cognito_user_id)
        
        with cache_lock: 
            cognito_cache[cognito_user_id] = {
                'settings': settings,
                'file_path': file_path,
                'data': data,
                'documents': documents,
                'vectors_path': vectors_path,
                'vectors': vectors,
                'index': index,
                'last_accessed': datetime.now()
            }
        return True
    except Exception as e:
        logger.error(f"Error in set_cognito_data: {e}")
        return None