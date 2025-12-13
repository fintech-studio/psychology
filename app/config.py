# 應用程式配置檔案

# 問卷設定
TOTAL_QUESTIONS = 8  # 問題總數，可以調整為任意數量

# 串流設定
STREAM_DELAY = 0.03  # 字元間隔時間（秒）

# Ollama local deployment settings (used to replace Gemini locally)
OLLAMA_API_URL = "http://172.25.1.24:11434"  # default ollama local API address
OLLAMA_MODEL_NAME = "llama3.1:8b"  # please set to your local model name
OLLAMA_ADVICE_TEMPERATURE = 0.7
OLLAMA_ADVICE_MAX_TOKENS = 1024
# Optional model fallbacks to try if the configured model
# cannot be found/reached
OLLAMA_MODEL_FALLBACKS = ["llama3.1:8b", "llama3.1:7b", "llama-3:8b"]

# 分析設定
ENABLE_CONTEXT_ANALYSIS = True  # 是否啟用上下文分析（問題+回答）
