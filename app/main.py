from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import time
import logging

# 導入應用模組
from routers.questionnaire_refactored import router as questionnaire_router
import models
from services import ollamaService

logging.basicConfig(level=logging.INFO)
# FastAPI 應用
app = FastAPI(title="心理問卷 API", version="1.0.0")
logger = logging.getLogger(__name__)

# CORS 中介軟體
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 開發環境，生產環境請限制 origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 調試中介軟體 - 記錄請求
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info("收到請求: %s %s", request.method, request.url)
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info("回應: %s - 耗時: %.3fs", response.status_code, process_time)
    return response

# 註冊路由
app.include_router(questionnaire_router)


@app.on_event("startup")
async def startup_event():
    """應用程式啟動時執行"""
    logger.info("正在載入分析模型...")
    try:
        # 初始化模型（延遲載入以加快 import/熱重載）
        models.init_models()
        if models.sentimentModel:
            logger.info("✅ 分析模型載入成功")
    except Exception as e:
        logger.exception("⚠️ 分析模型載入失敗: %s", e)

    # Initialize async services (e.g., health check for local LLM)
    try:
        await ollamaService.init()
    except Exception as e:
        logger.exception("⚠️ 初始化 OllamaService 時發生錯誤: %s", e)
    logger.info("🚀 心理問卷 API 啟動完成")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on application shutdown"""
    try:
        await ollamaService.shutdown()
        logger.info("OllamaService http client closed")
    except Exception:
        logger.exception("Failed to shutdown OllamaService cleanly")
    logger.info("心理問卷 API 已關閉")


@app.get("/")
def root():
    """根路徑"""
    return {
        "message": "心理問卷 API 服務",
        "version": "1.0.0",
        "endpoints": [
            "/questionnaire/start",
            "/questionnaire/answer",
            "/questionnaire/stream-question",
            "/questionnaire/save-question",
        ],
    }


@app.get("/health")
def health_check():
    """健康檢查端點"""
    return {
        "status": "healthy",
        "service": "psychology-questionnaire-api",
        "llm_available": ollamaService.is_api_available(),
        "llm_model": getattr(ollamaService, "model_name", None),
    }
