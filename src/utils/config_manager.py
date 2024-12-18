import threading
from typing import Dict, Any
from src.utils.file_utils import update_settings_path
import json
from src.utils.logging_config import logger

# グローバルロックの作成
settings_lock = threading.Lock()
user_settings: Dict[str, Any] = {}

# デフォルト設定の定義
DEFAULT_SETTINGS = {
    'max_requests': float('inf'),
    'reset_time': 3600,
    'threshold': 0.7,
    'model': 'gpt-3.5-turbo-0125',
    'knowledge_about_user': "",
    'response_preference': "",
    'log_option': 'off',
    'history_maxlen': 12,
    'USERNAME': None,  # 初期値をNoneに設定
    'PASSWORD': None,  # 初期値をNoneに設定
    'questions': [],
    'corresponding_ids': []
}

from typing import Dict, Any

def load_settings(cognito_user_id: str) -> Dict[str, Any]:
    settings_path = update_settings_path(cognito_user_id)
    try:
        with open(settings_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)
        
        # デフォルト値で辞書を初期化し、読み込んだ設定で上書き
        result = DEFAULT_SETTINGS.copy()
        result.update(settings)
        
        # 特別な処理が必要な項目の処理
        if result['max_requests'] == "Infinity":
            result['max_requests'] = float('inf')
        
        # 文字列をリストに変換
        result['questions'] = result['questions'].split("\n") if result['questions'] else []
        result['corresponding_ids'] = result['corresponding_ids'].split("\n") if result['corresponding_ids'] else []
        
        return result
    except FileNotFoundError:
        return DEFAULT_SETTINGS.copy()
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in settings file for user {cognito_user_id}")
        return DEFAULT_SETTINGS.copy()

def get_user_setting(cognito_user_id, key):
    with settings_lock:
        return user_settings.get(cognito_user_id, {}).get(key, DEFAULT_SETTINGS.get(key))