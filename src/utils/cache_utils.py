from datetime import datetime, timedelta
import threading
import time
from typing import Dict, Any
from src.utils.logging_config import logger

# キャッシュの有効期限（秒）
CACHE_EXPIRY = 3600
# キャッシュ用の辞書
cognito_cache: Dict[str, Any] = {}
# グローバルロックの作成
cache_lock = threading.Lock()

def check_cache_expiry():
    while True:
        with cache_lock:
            current_time = datetime.now()
            keys_to_delete = []
            for cognito_user_id, cache_data in cognito_cache.items():
                if current_time - cache_data['last_accessed'] > timedelta(seconds=CACHE_EXPIRY):
                    keys_to_delete.append(cognito_user_id)
            for key in keys_to_delete:
                del cognito_cache[key]
                logger.info(f"Cache for Cognito user ID {key} has been cleared.")
        time.sleep(CACHE_EXPIRY)