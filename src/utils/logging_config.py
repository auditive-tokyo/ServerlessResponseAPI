import logging
import sys

# loggerの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# StreamHandlerを使用してコンソールに出力
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)

# フォーマットの設定
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)