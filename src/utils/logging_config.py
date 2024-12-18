import logging
from logging.handlers import RotatingFileHandler

# ロガーの設定
logger = logging.getLogger('app_logger')
logger.setLevel(logging.INFO)

# ハンドラの設定
handler = RotatingFileHandler('app.log', maxBytes=5*1024*1024, backupCount=10)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)