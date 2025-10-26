from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from schemas.questionnaire import StartResponse, AnswerRequest, NextQuestionResponse, StreamQuestionRequest, SaveQuestionRequest
from config import TOTAL_QUESTIONS
from typing import Dict, Any
import json
from services import analysisService, geminiService, questionnaireService

router = APIRouter(prefix="/questionnaire", tags=["questionnaire"])

@router.post("/start", response_model=StartResponse)
async def start_questionnaire() -> StartResponse:
    """開始問卷調查"""
    try:
        session_id = questionnaireService.create_session()

        # 動態生成第一個問題
        first_question = await geminiService.generate_dynamic_question(
            current_number=1, 
            total_questions=TOTAL_QUESTIONS, 
            previous_responses=None
        )
        
        # 保存生成的問題
        questionnaireService.save_generated_question(session_id, first_question)
        
        return StartResponse(
            session_id=session_id,
            question=first_question,
            question_number=1,
            total_questions=TOTAL_QUESTIONS
        )
    except Exception as e:
        print(f"開始問卷時發生錯誤: {e}")
        raise HTTPException(status_code=500, detail="伺服器內部錯誤")

@router.post("/answer", response_model=NextQuestionResponse)
async def submit_answer(request: AnswerRequest) -> NextQuestionResponse:
    """提交答案並取得下一個問題"""
    try:
        # 取得當前問題（用於分析）
        current_question = questionnaireService.get_current_question(request.session_id)
        if not current_question:
            raise HTTPException(status_code=404, detail="會話不存在或已完成")
        
        # 分析使用者回應
        sentiment_scores, stress_scores = analysisService.analyze_user_response(
            request.answer, 
            current_question
        )
        
        # 儲存回答
        success = questionnaireService.save_response(
            request.session_id, 
            request.answer, 
            sentiment_scores, 
            stress_scores
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="儲存回答失敗")
        
        # 檢查是否還有下一個問題
        if questionnaireService.is_questionnaire_complete(request.session_id):
            # 問卷完成，生成最終建議
            all_responses = questionnaireService.get_all_responses(request.session_id)
            advice = await geminiService.generate_content(all_responses)

            return NextQuestionResponse(
                has_next_question=False,
                advice=advice
            )
        else:
            # 需要生成下一個問題
            progress = questionnaireService.get_progress(request.session_id)
            all_responses = questionnaireService.get_all_responses(request.session_id)

            next_question = await geminiService.generate_dynamic_question(
                current_number=progress["current"] + 1,
                total_questions=progress["total"],
                previous_responses=all_responses
            )
            
            # 保存生成的問題
            questionnaireService.save_generated_question(request.session_id, next_question)
            
            return NextQuestionResponse(
                has_next_question=True,
                question=next_question,
                question_number=progress["current"] + 1,
                total_questions=progress["total"]
            )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"提交答案時發生錯誤: {e}")
        raise HTTPException(status_code=500, detail="伺服器內部錯誤")

@router.post("/stream-question")
async def stream_question(request: StreamQuestionRequest):
    """串流顯示問題"""
    try:
        # 取得當前問題
        current_question = questionnaireService.get_current_question(request.session_id)

        if not current_question:
            raise HTTPException(status_code=404, detail="會話不存在或問題不存在")

        progress = questionnaireService.get_progress(request.session_id)
        all_responses = questionnaireService.get_all_responses(request.session_id)
        
        async def generate_stream():
            async for chunk in geminiService.stream_question_generation(
                progress["current"] + 1,
                TOTAL_QUESTIONS,
                all_responses
            ):
                # 如果問題生成完成，保存問題到會話
                if chunk.get("done") and chunk.get("question"):
                    questionnaireService.save_generated_question(request.session_id, chunk["question"])
                
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"串流問題時發生錯誤: {e}")
        raise HTTPException(status_code=500, detail="伺服器內部錯誤")

@router.post("/save-question")
async def save_question(request: SaveQuestionRequest) -> Dict[str, Any]:
    """儲存問題回答"""
    try:
        # 取得當前問題
        current_question = questionnaireService.get_current_question(request.session_id)
        if not current_question:
            raise HTTPException(status_code=404, detail="會話不存在或已完成")
        
        # 分析使用者回應
        sentiment_scores, stress_scores = analysisService.analyze_user_response(
            request.answer, 
            current_question
        )
        
        # 儲存回答
        success = questionnaireService.save_response(
            request.session_id, 
            request.answer, 
            sentiment_scores, 
            stress_scores
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="儲存回答失敗")
        
        # 檢查是否完成問卷
        is_complete = questionnaireService.is_questionnaire_complete(request.session_id)
        
        if is_complete:
            # 問卷完成，生成最終建議
            all_responses = questionnaireService.get_all_responses(request.session_id)
            advice = await geminiService.generate_content(all_responses)
            
            return {
                "success": True,
                "is_complete": True,
                "advice": advice
            }
        else:
            # 還有下一個問題
            progress = questionnaireService.get_progress(request.session_id)
            return {
                "success": True,
                "is_complete": False,
                "next_question_number": progress["current"] + 1,
                "total_questions": progress["total"]
            }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"儲存問題時發生錯誤: {e}")
        raise HTTPException(status_code=500, detail="伺服器內部錯誤")