# Stub: StressModel is disabled; keep a minimal interface to avoid calling errors
import logging

logger = logging.getLogger(__name__)


class StressModel:
    def __init__(self):
        # Do not load heavy models; warn via logger instead of printing
        logger.warning("StressModel disabled: stress analysis removed from the project.")

    def analyze(self, text_zh):
        """
        回傳空的分析結果結構（與原有模型回傳格式相容的最小表現）。
        這樣呼叫端仍可處理返回值而不會拋例外。
        """
        return []  # Return an empty list to preserve original structure
