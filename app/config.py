# 應用程式配置檔案

# 問卷設定
TOTAL_QUESTIONS = 4  # 問題總數，可以調整為任意數量
MIN_QUESTIONS = 3    # 最少問題數
MAX_QUESTIONS = 10   # 最多問題數

# 串流設定
STREAM_DELAY = 0.03  # 字元間隔時間（秒）

# API 設定
GEMINI_MODEL_NAME = "gemini-2.0-flash"
GEMINI_TEMPERATURE = 0.8
GEMINI_MAX_TOKENS = 150
GEMINI_ADVICE_TEMPERATURE = 0.7
GEMINI_ADVICE_MAX_TOKENS = 1024

# 分析設定
ENABLE_CONTEXT_ANALYSIS = True  # 是否啟用上下文分析（問題+回答）