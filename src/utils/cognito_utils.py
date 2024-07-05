from datetime import datetime
from src.schema.logging_config import logger
from src.utils.cache_utils import cache_lock, cognito_cache
from src.functions.vector_handling import load_vectors_and_create_index
from src.utils.file_utils import update_settings_path, get_file_path
from src.schema.config_manager import load_config

def set_cognito_data(cognito_user_id):
    try:
        # settings_pathを更新
        update_settings_path(cognito_user_id)
        # 設定をロード
        load_config(cognito_user_id)
        # reference.jsonのpath更新
        file_path, data, documents = get_file_path(cognito_user_id)
        # vectors_pathの更新
        vectors_path, vectors, index = load_vectors_and_create_index(cognito_user_id)
    except Exception as e:
        logger.error(f"Error in set_cognito_data: {e}")
        return None

    with cache_lock: 
        cognito_cache[cognito_user_id] = {
            'file_path': file_path,
            'data': data,
            'documents': documents,
            'vectors_path': vectors_path,
            'vectors': vectors,
            'index': index,
            'last_accessed': datetime.now()
        }
    return True