# 心理問卷 API 服務

基於 FastAPI 和 AI 模型的智能心理問卷系統，用於分析用戶的投資心理特徵和風險偏好。

## 功能特色

- 🤖 **AI 驅動問題生成**：使用 Ollama 本地 LLM 動態生成個性化問題
- 📊 **多維度分析**：情緒分析、壓力評估、風險偏好、決策模式分析
- 🎯 **投資者分類**：基於心理特徵自動分類投資者類型
- 📈 **結構化報告**：生成詳細的心理分析報告和建議

## 技術架構

### 核心技術

- **FastAPI**: 高性能 Web 框架
- **Ollama**: 本地 LLM 服務
- **Transformers**: Hugging Face 模型庫
- **FinBERT**: 金融領域情感分析模型
- **PyTorch**: 深度學習框架

### 專案結構

```
app/
├── models/                         # 模型
│   ├── SentimentModel.py           # 情感分析模型
│   └── StressModel.py              # 壓力評估模型
├── routers/                        # API 路由
│   └── questionnaire_refactored.py
├── schemas/                        # 資料結構定義
│   └── questionnaire.py
├── services/                       # 業務邏輯服務
│   ├── analysis_service.py         # 分析服務
│   ├── ollama_service.py           # Ollama LLM 服務
│   ├── questionnaire_service.py    # 問卷服務
│   └── prompt_templates.py         # 提示詞模板
├── utils/                          # 工具函數
│   └── Translate.py                # 翻譯工具
├── config.py                       # 配置文件
└── main.py                         # 應用入口
```

## 快速開始

### 環境需求

- Python 3.12+
- Ollama (本地 LLM 服務)
- CUDA

### 安裝步驟

1. **clone 專案**

   ```bash
   git clone <repository-url>
   cd psychology
   ```

2. **安裝依賴**

   ```bash
   # 使用 pip 安裝所需套件
   pip install -r requirements.txt

   # 或使用 uv
   uv sync
   ```

3. **配置 Ollama**

   ```bash
   # 安裝並啟動 Ollama
   ollama serve

   # 下載模型
   ollama pull llama3.1:8b
   ```

4. **配置環境**
   編輯 `app/config.py` 設定 Ollama 服務地址：

   ```python
   OLLAMA_API_URL = "http://localhost:11434"
   OLLAMA_MODEL_NAME = "llama3.1:8b"
   ```

   或使用 `.env` 文件：

   ```
   OLLAMA_API_URL="http://localhost:11434"
   OLLAMA_MODEL_NAME="llama3.1:8b"
   ```

5. **啟動服務**

   ```bash
   cd app
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

## API 端點

### 核心功能

- `POST /questionnaire/start` - 開始問卷
- `POST /questionnaire/answer` - 提交答案
- `POST /questionnaire/stream-question` - 串流生成問題
- `POST /questionnaire/save-question` - 儲存問題與答案

### 系統端點

- `GET /` - 服務資訊
- `GET /health` - 健康檢查

## 分析維度

系統會從以下六個維度分析用戶心理特徵：

1. **情緒 (Emotion)** - 情緒穩定性與反應模式
2. **壓力 (Stress)** - 壓力承受能力與應對方式
3. **風險 (Risk)** - 風險偏好與容忍度
4. **決策 (Decision)** - 決策風格與衝動控制
5. **時間 (Time)** - 投資時間偏好
6. **一般 (General)** - 綜合心理特徵

## 投資者類型

基於分析結果，系統會將用戶分類為：

- **保守型投資者** - 風險厭惡，偏好穩定收益
- **平衡型投資者** - 風險中性，追求平衡配置
- **積極型投資者** - 風險偏好，追求高收益
