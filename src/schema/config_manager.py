import threading
import json
import openai
from typing import Dict, Any
from src.schema.logging_config import logger
from src.utils.file_utils import update_settings_path

# グローバルロックの作成
settings_lock = threading.Lock()
user_settings: Dict[str, Any] = {}

# 初期値の定義
DEFAULT_MAX_REQUESTS = float('inf')
DEFAULT_RESET_TIME = 3600
DEFAULT_THRESHOLD = 0.7
DEFAULT_MODEL = 'gpt-3.5-turbo-0125'
DEFAULT_KNOWLEDGE_ABOUT_USER = ""
DEFAULT_RESPONSE_PREFERENCE = ""
DEFAULT_LOG_OPTION = 'off'
DEFAULT_HISTORY_MAXLEN = 12
DEFAULT_USERNAME = 'admin'
DEFAULT_PASSWORD = 'password'

def load_config(cognito_user_id):
    with settings_lock:
        try:
            settings_path = update_settings_path(cognito_user_id)
            with open(settings_path, 'r') as f:
                config = json.load(f)
                user_settings[cognito_user_id] = {
                    'api_key': config.get('api_key', openai.api_key),
                    'max_requests': float(config.get('max_requests', DEFAULT_MAX_REQUESTS)) if config.get('max_requests') != "Infinity" else float('inf'),
                    'reset_time': int(config.get('reset_time', DEFAULT_RESET_TIME)),
                    'threshold': float(config.get('threshold', DEFAULT_THRESHOLD)),
                    'model': config.get('model', DEFAULT_MODEL),
                    'knowledge_about_user': config.get('knowledge_about_user', DEFAULT_KNOWLEDGE_ABOUT_USER),
                    'response_preference': config.get('response_preference', DEFAULT_RESPONSE_PREFERENCE),
                    'log_option': config.get('log_option', DEFAULT_LOG_OPTION),
                    'history_maxlen': int(config.get('history_maxlen', DEFAULT_HISTORY_MAXLEN)) if config.get('history_maxlen', DEFAULT_HISTORY_MAXLEN) != float('inf') else float('inf'),
                    'USERNAME': config.get('USERNAME', DEFAULT_USERNAME),
                    'PASSWORD': config.get('PASSWORD', DEFAULT_PASSWORD),
                    'questions': config.get('questions', '').split("\n") if config.get('questions', '') else [],
                    'corresponding_ids': config.get('corresponding_ids', '').split("\n") if config.get('corresponding_ids', '') else []
                }
        except FileNotFoundError:
            pass  # ファイルが見つからない場合は特に何もしない
        except Exception as e:
            logger.error(f"Error in load_config: {e}")