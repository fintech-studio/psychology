from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import time
import os
import sys

# 添加專案根目錄到 Python 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 載入環境變數
from dotenv import load_dotenv
load_dotenv()

# 導入應用模組
from app.routers.questionnaire import router as questionnaire_router
from app.services.analysis_service import AnalysisService

# FastAPI 應用
app = FastAPI(title="心理問卷 API", version="1.0.0")

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
    print(f"收到請求: {request.method} {request.url}")
    response = await call_next(request)
    process_time = time.time() - start_time
    print(f"回應: {response.status_code} - 耗時: {process_time:.3f}s")
    return response

# 註冊路由
app.include_router(questionnaire_router)

@app.on_event("startup")
async def startup_event():
    """應用程式啟動時執行"""
    print("正在載入分析模型...")
    
    # 初始化服務
    analysis_service = AnalysisService()
    
    try:
        analysis_service.ensure_models_loaded()
        print("✅ 分析模型載入成功")
    except Exception as e:
        print(f"⚠️  分析模型載入失敗: {e}")
    
    print(f"🚀 心理問卷 API 啟動完成")

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
            "/questionnaire/save-question"
        ]
    }

@app.get("/health")
def health_check():
    """健康檢查端點"""
    return {"status": "healthy", "service": "psychology-questionnaire-api"}

# if __name__ == "__main__":
#     import uvicorn
#     print("🚀 啟動心理問卷 API 服務...")
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)