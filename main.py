from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Dict, Any, List, Optional
import json
import asyncio
import os
import uuid
import threading

# 載入環境變數
from dotenv import load_dotenv
load_dotenv()

# Google Generative AI SDK
import google.generativeai as genai

# 本地模型匯入
from models.SentimentModel import SentimentModel
from models.StressModel import StressModel

# === 配置 ===
TOTAL_QUESTIONS = 3  # 總問題數
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-2.0-flash"

# 配置 Gemini
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    print("警告：未設定 GOOGLE_API_KEY")

# FastAPI 應用
app = FastAPI(title="理財問卷 API", version="1.0.0")

# CORS 中介軟體
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 開發環境，生產環境請限制 origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全域變數
SESSIONS: Dict[str, Dict[str, Any]] = {}
SESSIONS_LOCK = threading.Lock()
SENTIMENT_MODEL: Optional[SentimentModel] = None
STRESS_MODEL: Optional[StressModel] = None

# === API 模型 ===
class StartResponse(BaseModel):
    session_id: str
    # 移除 question，改由串流端點提供

class AnswerRequest(BaseModel):
    session_id: str
    answer_text: str

class NextQuestionResponse(BaseModel):
    next_question: Optional[str] = None
    finished: bool = False
    advice: Optional[str] = None

class StreamQuestionRequest(BaseModel):
    session_id: str
    question_number: int

# === Gemini 相關函式 ===
async def stream_question_generation(prompt: str):
    """串流方式生成問題"""
    if not GOOGLE_API_KEY:
        # Mock 串流回應 - 模擬問題生成
        import random
        mock_questions = [
            "請簡單描述您的投資經驗？",
            "面對投資虧損時您會如何反應？", 
            "您投資的主要目標是什麼？",
            "您每月可投資的金額大約多少？",
            "您偏好哪種投資方式？"
        ]
        question = random.choice(mock_questions)
        for char in question:
            yield f"data: {json.dumps({'text': char, 'done': False}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.05)  # 控制顯示速度
        yield f"data: {json.dumps({'text': '', 'done': True}, ensure_ascii=False)}\n\n"
        return
    
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.7,
                max_output_tokens=1024,
            ),
            stream=True  # 啟用串流
        )
        
        for chunk in response:
            if chunk.text:
                # 逐字發送
                for char in chunk.text:
                    yield f"data: {json.dumps({'text': char, 'done': False}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.03)  # 控制顯示速度
        
        # 發送完成信號
        yield f"data: {json.dumps({'text': '', 'done': True}, ensure_ascii=False)}\n\n"
        
    except Exception as e:
        print(f"Gemini 串流 API 錯誤: {e}")
        error_msg = "(系統發生錯誤，無法取得建議)"
        for char in error_msg:
            yield f"data: {json.dumps({'text': char, 'done': False}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.05)
        yield f"data: {json.dumps({'text': '', 'done': True}, ensure_ascii=False)}\n\n"

def call_gemini(prompt: str, is_question: bool = False) -> str:
    """呼叫 Gemini 生成內容"""
    if not GOOGLE_API_KEY:
        # Mock 回應 - 提供基本的問題輪替
        if is_question:
            import random
            mock_questions = [
                "請簡單描述您的投資經驗？",
                "面對投資虧損時您會如何反應？", 
                "您投資的主要目標是什麼？",
                "您每月可投資的金額大約多少？",
                "您偏好哪種投資方式？"
            ]
            return random.choice(mock_questions)
        else:
            return "根據您的情緒與壓力分析，建議：保持多樣化投資組合、設定適當停損點、降低投資槓桿，並尋求專業理財諮詢。"
    
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.7,
                max_output_tokens=1024,
            )
        )
        
        if response.text:
            return response.text.strip()
        else:
            return "(系統暫時無法生成回應，請稍後再試)"
            
    except Exception as e:
        print(f"Gemini API 錯誤: {e}")
        if "quota" in str(e).lower():
            return "(API 配額已用完，請稍後再試)"
        elif "permission" in str(e).lower():
            return "(API 金鑰權限不足，請檢查設定)"
        else:
            return "(系統發生錯誤，無法取得建議)"

def generate_question(question_number: int, previous_answers: List[str] = None) -> str:
    """動態生成投資理財相關問題"""
    # 基礎問題提示詞，聚焦於投資心理和情緒
    base_prompts = [
        "請生成一個關於投資時心理壓力的簡短問題（15-25字），目的:了解使用者投資時的壓力感受。",
        "請生成一個關於投資情緒波動的簡短問題（15-25字），目的:了解使用者面對市場變化時的情緒反應。",
        "請生成一個關於投資焦慮的簡短問題（15-25字），目的:了解使用者在投資過程中的擔憂和恐懼。",
        "請生成一個關於投資決策心理的簡短問題（15-25字），目的:了解使用者做投資決定時的心理狀態。",
        "請生成一個關於投資壓力管理的簡短問題（15-25字），目的:了解使用者如何處理投資帶來的心理負擔。"
    ]
    
    # 如果有前面的回答，加入上下文來生成更針對性的問題
    if previous_answers and question_number > 0:
        context = ""
        for i, answer in enumerate(previous_answers):
            context += f"問題{i+1}回答：{answer[:50]}{'...' if len(answer) > 50 else ''}\n"
        
        if question_number < len(base_prompts):
            prompt = f"""
基於使用者之前的回答：
{context}

{base_prompts[question_number]}
請確保：
1. 問題長度控制在15-25字
2. 避免與之前問題重複
3. 問題要具體且易於回答
4. 只回傳問題本身，不要其他說明
"""
        else:
            # 超過基礎問題時，根據前面回答生成深度心理問題
            prompt = f"""
基於使用者之前的回答：
{context}

請生成一個深入的投資心理問題（15-25字），專注了解使用者的：
- 投資時的內心感受和情緒
- 面對虧損時的心理反應
- 投資壓力的來源和影響
- 情緒如何影響投資決策
- 投資焦慮和恐懼的具體表現

要求：
1. 問題長度15-25字
2. 聚焦心理和情緒層面，不要問金額或策略
3. 避免重複之前的內容
4. 只回傳問題本身
"""
    else:
        # 第一個問題或沒有上下文時使用基礎提示
        if question_number < len(base_prompts):
            prompt = base_prompts[question_number] + "\n要求：只回傳問題本身，長度15-25字。"
        else:
            prompt = "請生成一個關於投資心理壓力或情緒的簡短問題（15-25字），了解使用者的投資心理狀態。只回傳問題本身。"
    
    return call_gemini(prompt, is_question=True)

# === 分析模型相關函式 ===
def sanitize_sentiment_output(raw) -> Dict[str, float]:
    """解析 SentimentModel 輸出，提取 negative、neutral、positive 分數"""
    result = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
    
    try:
        # 處理巢狀列表格式: [[{...}]]
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            data = raw[0]
        elif isinstance(raw, list):
            data = raw
        else:
            return result
            
        for item in data:
            if not isinstance(item, dict):
                continue
                
            label = item.get("label", "").lower()
            score = float(item.get("score", 0.0))
            
            if "negative" in label or "neg" in label:
                result["negative"] = score
            elif "positive" in label or "pos" in label:
                result["positive"] = score
            elif "neutral" in label or "neu" in label:
                result["neutral"] = score
                
    except Exception as e:
        print(f"解析情緒輸出時發生錯誤: {e}")
        
    return result

def sanitize_stress_output(raw) -> Dict[str, float]:
    """解析 StressModel 輸出，提取 stress、not_stress 分數"""
    result = {"stress": 0.0, "not_stress": 0.0}
    
    try:
        # 處理巢狀列表格式: [[{...}]]
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            data = raw[0]
        elif isinstance(raw, list):
            data = raw
        else:
            return result
            
        for item in data:
            if not isinstance(item, dict):
                continue
                
            label = item.get("label", "").lower()
            score = float(item.get("score", 0.0))
            
            if "stressed" in label and "not" not in label:
                result["stress"] = score
            elif "not stressed" in label or "not" in label:
                result["not_stress"] = score
                
    except Exception as e:
        print(f"解析壓力輸出時發生錯誤: {e}")
        
    return result

def analyze_user_response(text: str, question: str = "") -> tuple[Dict[str, float], Dict[str, float]]:
    """分析使用者回應，回傳情緒和壓力分數"""
    global SENTIMENT_MODEL, STRESS_MODEL
    
    # 確保模型已載入
    if SENTIMENT_MODEL is None:
        SENTIMENT_MODEL = SentimentModel()
    if STRESS_MODEL is None:
        STRESS_MODEL = StressModel()
    
    # 組合問題和回答提供完整上下文
    if question.strip():
        # 格式: "問題：{question} 回答：{text}" 
        analysis_text = f"問題：{question.strip()} 回答：{text.strip()}"
    else:
        # 如果沒有問題，就直接用回答
        analysis_text = text.strip()
    
    # 執行分析（使用包含上下文的文本）
    sentiment_raw = SENTIMENT_MODEL.analyze(analysis_text)
    stress_raw = STRESS_MODEL.analyze(analysis_text)
    
    # 解析結果
    sentiment_scores = sanitize_sentiment_output(sentiment_raw)
    stress_scores = sanitize_stress_output(stress_raw)
    
    return sentiment_scores, stress_scores

def generate_final_advice(all_sentiment_scores: List[Dict[str, float]], 
                         all_stress_scores: List[Dict[str, float]]) -> str:
    """根據所有問題的情緒與壓力分數生成最終建議"""
    
    # 計算平均分數
    avg_sentiment = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
    avg_stress = {"stress": 0.0, "not_stress": 0.0}
    
    if all_sentiment_scores:
        for key in avg_sentiment:
            avg_sentiment[key] = sum(s.get(key, 0) for s in all_sentiment_scores) / len(all_sentiment_scores)
    
    if all_stress_scores:
        for key in avg_stress:
            avg_stress[key] = sum(s.get(key, 0) for s in all_stress_scores) / len(all_stress_scores)
    
    # 建立分析摘要
    summary_lines = []
    for i, (sentiment, stress) in enumerate(zip(all_sentiment_scores, all_stress_scores), 1):
        summary_lines.append(
            f"問題{i}: 負面={sentiment.get('negative', 0):.3f}, "
            f"中性={sentiment.get('neutral', 0):.3f}, "
            f"正面={sentiment.get('positive', 0):.3f}, "
            f"壓力={stress.get('stress', 0):.3f}, "
            f"無壓力={stress.get('not_stress', 0):.3f}"
        )
    
    # 生成 prompt 給 Gemini
    prompt = f"""
請根據以下使用者在投資理財問卷中的情緒與壓力分析結果，提供個人化的理財建議與心理健康建議：

詳細分析結果：
{chr(10).join(summary_lines)}

平均情緒分數：
- 負面情緒: {avg_sentiment['negative']:.3f}
- 中性情緒: {avg_sentiment['neutral']:.3f}  
- 正面情緒: {avg_sentiment['positive']:.3f}

平均壓力指標：
- 有壓力: {avg_stress['stress']:.3f}
- 無壓力: {avg_stress['not_stress']:.3f}

請提供：
1. 投資心理狀態分析
2. 個人化理財建議（包含風險管理、資產配置等）
3. 壓力管理與情緒調適建議
4. 具體的行動方案

請用繁體中文回答，內容要實用且易於執行。
注意：請不要使用任何 Markdown 格式標記（如 ** 粗體標記），回答內容應該是純文字格式。
    """
    
    advice_text = call_gemini(prompt, is_question=False)
    
    # 移除所有的 Markdown 格式標記
    clean_advice = advice_text.replace("**", "").replace("*", "")
    
    return clean_advice

# === API 端點 ===
@app.get("/models")
def get_models_info():
    """取得模型資訊（除錯用）"""
    return {
        "model_name": MODEL_NAME,
        "api_key_set": bool(GOOGLE_API_KEY),
        "total_questions": TOTAL_QUESTIONS
    }

@app.post("/start", response_model=StartResponse)
def start_test():
    """開始新測驗"""
    session_id = str(uuid.uuid4())
    
    # 建立 session（不生成第一個問題，改由串流端點處理）
    with SESSIONS_LOCK:
        SESSIONS[session_id] = {
            "question_index": 0,
            "questions": [],
            "answers": [],
            "sentiment_scores": [],
            "stress_scores": [],
        }
    
    return StartResponse(session_id=session_id)

@app.post("/answer", response_model=NextQuestionResponse)
def submit_answer(req: AnswerRequest):
    """提交回答"""
    # 驗證 session
    with SESSIONS_LOCK:
        if req.session_id not in SESSIONS:
            raise HTTPException(status_code=404, detail="Session 不存在")
        session = SESSIONS[req.session_id]
        
        # 取得當前問題（用於提供上下文給模型分析）
        current_question = ""
        if session["question_index"] < len(session["questions"]):
            current_question = session["questions"][session["question_index"]]
    
    # 分析使用者回應（包含問題上下文）
    sentiment_scores, stress_scores = analyze_user_response(req.answer_text, current_question)
    
    # 更新 session
    with SESSIONS_LOCK:
        session["answers"].append(req.answer_text)
        session["sentiment_scores"].append(sentiment_scores)
        session["stress_scores"].append(stress_scores)
        session["question_index"] += 1
        
        current_question_index = session["question_index"]
    
    # 檢查是否完成所有問題
    if current_question_index >= TOTAL_QUESTIONS:
        # 生成最終建議
        advice = generate_final_advice(session["sentiment_scores"], session["stress_scores"])
        
        # 清理 session
        with SESSIONS_LOCK:
            del SESSIONS[req.session_id]
        
        return NextQuestionResponse(
            next_question=None,
            finished=True,
            advice=advice
        )
    else:
        # 生成下一個問題（帶有之前回答的上下文）
        next_question = generate_question(current_question_index, session["answers"])
        
        # 更新 session
        with SESSIONS_LOCK:
            session["questions"].append(next_question)
        
        return NextQuestionResponse(
            next_question=next_question,
            finished=False,
            advice=None
        )

@app.post("/stream-question")
async def stream_question(req: StreamQuestionRequest):
    """串流生成問題"""
    # 驗證 session
    with SESSIONS_LOCK:
        if req.session_id not in SESSIONS:
            raise HTTPException(status_code=404, detail="Session 不存在")
        session = SESSIONS[req.session_id]
        previous_answers = session.get("answers", [])
    
    # 根據問題編號和之前的回答生成問題
    question_number = req.question_number
    
    # 建立 prompt
    base_prompts = [
        "請生成一個關於投資時心理壓力的簡短問題（15-25字），目的:了解使用者投資時的壓力感受。",
        "請生成一個關於投資情緒波動的簡短問題（15-25字），目的:了解使用者面對市場變化時的情緒反應。",
        "請生成一個關於投資焦慮的簡短問題（15-25字），目的:了解使用者在投資過程中的擔憂和恐懼。",
        "請生成一個關於投資決策心理的簡短問題（15-25字），目的:了解使用者做投資決定時的心理狀態。",
        "請生成一個關於投資壓力管理的簡短問題（15-25字），目的:了解使用者如何處理投資帶來的心理負擔。"
    ]
    
    if previous_answers and question_number > 0:
        context = ""
        for i, answer in enumerate(previous_answers):
            context += f"問題{i+1}回答：{answer[:50]}{'...' if len(answer) > 50 else ''}\n"
        
        if question_number < len(base_prompts):
            prompt = f"""
基於使用者之前的回答：
{context}

{base_prompts[question_number]}
請確保：
1. 問題長度控制在15-25字
2. 避免與之前問題重複
3. 問題要具體且易於回答
4. 只回傳問題本身，不要其他說明
"""
        else:
            prompt = f"""
基於使用者之前的回答：
{context}

請生成一個深入的投資心理問題（15-25字），專注了解使用者的：
- 投資時的內心感受和情緒
- 面對虧損時的心理反應
- 投資壓力的來源和影響
- 情緒如何影響投資決策
- 投資焦慮和恐懼的具體表現

要求：
1. 問題長度15-25字
2. 聚焦心理和情緒層面，不要問金額或策略
3. 避免重複之前的內容
4. 只回傳問題本身
"""
    else:
        if question_number < len(base_prompts):
            prompt = base_prompts[question_number] + "\n要求：只回傳問題本身，長度15-25字。"
        else:
            prompt = "請生成一個關於投資心理壓力或情緒的簡短問題（15-25字），了解使用者的投資心理狀態。只回傳問題本身。"
    
    # 回傳串流問題
    return StreamingResponse(
        stream_question_generation(prompt),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.post("/save-question")
def save_question(req: dict):
    """保存串流生成的問題到 session"""
    session_id = req.get("session_id")
    question = req.get("question")
    
    if not session_id or not question:
        raise HTTPException(status_code=400, detail="缺少必要參數")
    
    with SESSIONS_LOCK:
        if session_id in SESSIONS:
            SESSIONS[session_id]["questions"].append(question)
            return {"message": "問題已保存"}
        else:
            raise HTTPException(status_code=404, detail="Session 不存在")

@app.on_event("startup")
async def startup_event():
    """應用程式啟動時執行"""
    global SENTIMENT_MODEL, STRESS_MODEL
    
    print("正在載入分析模型...")
    
    try:
        if SENTIMENT_MODEL is None:
            SENTIMENT_MODEL = SentimentModel()
            print("✅ SentimentModel 載入成功")
    except Exception as e:
        print(f"⚠️  SentimentModel 載入失敗: {e}")
    
    try:
        if STRESS_MODEL is None:
            STRESS_MODEL = StressModel()
            print("✅ StressModel 載入成功")
    except Exception as e:
        print(f"⚠️  StressModel 載入失敗: {e}")
    
    print(f"🚀 理財問卷 API 啟動完成 (模型: {MODEL_NAME})")

@app.get("/")
def root():
    """根路徑"""
    return {"message": "理財問卷 API 服務", "version": "1.0.0", "endpoints": ["/start", "/answer", "/models"]}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)