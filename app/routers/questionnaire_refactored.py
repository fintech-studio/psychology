from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from schemas.questionnaire import (
    StartResponse,
    AnswerRequest,
    NextQuestionResponse,
    StreamQuestionRequest,
    SaveQuestionRequest,
)
from config import TOTAL_QUESTIONS
from typing import Dict, Any, Optional
import json
import logging
from services import analysisService, ollamaService, questionnaireService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/questionnaire", tags=["questionnaire"])


async def _get_current_question_or_404(session_id: str) -> str:
    current_question = questionnaireService.get_current_question(session_id)
    if not current_question:
        raise HTTPException(status_code=404, detail="會話不存在或問題不存在")
    return current_question


async def _generate_and_save_next(
    session_id: str, target_number: int, previous_responses: Optional[list]
):
    next_question, next_meta = await ollamaService.generate_dynamic_question(
        current_number=target_number,
        total_questions=TOTAL_QUESTIONS,
        previous_responses=previous_responses,
    )
    questionnaireService.save_generated_question(
        session_id,
        next_question,
        index=target_number - 1,
        meta=(next_meta if next_meta else None),
    )
    return next_question, next_meta


async def _finalize_session(session_id: str):
    all_responses = questionnaireService.get_all_responses(session_id)
    try:
        analysis = await ollamaService.generate_content(all_responses)
        if isinstance(analysis, dict):
            advice = analysis.get("advice")
        else:
            advice = str(analysis)
    except Exception as e:
        logger.exception("生成建議時發生錯誤: %s", e)
        raise HTTPException(
            status_code=503,
            detail=(
                "AI model not available to generate advice"
            ),
        )

    profile = analysisService.compute_profile(all_responses)
    investor_type = analysisService.classify_investor(profile)
    return advice, analysis, profile, investor_type


@router.post("/start", response_model=StartResponse)
async def start_questionnaire() -> StartResponse:
    try:
        session_id = questionnaireService.create_session()
        logger.info("新會話建立：%s", session_id)
        try:
            (first_question, first_meta) = (
                await ollamaService.generate_dynamic_question(
                    current_number=1,
                    total_questions=TOTAL_QUESTIONS,
                    previous_responses=None,
                )
            )
        except Exception as e:
            logger.exception("在生成第一題時發生錯誤: %s", e)
            raise HTTPException(
                status_code=503,
                detail=(
                    "AI model not available to generate question"
                ),
            )
        saved = questionnaireService.save_generated_question(
            session_id,
            first_question,
            index=0,
            meta=(first_meta if first_meta else None),
        )
        if not saved:
            logger.warning("無法儲存第一題到會話 %s", session_id)
        return StartResponse(
            session_id=session_id,
            question=first_question,
            question_number=1,
            total_questions=TOTAL_QUESTIONS,
            question_meta=(first_meta if first_meta else None),
        )
    except Exception as e:
        logger.exception("開始問卷時發生錯誤: %s", e)
        raise HTTPException(status_code=500, detail="伺服器內部錯誤")


@router.post("/answer", response_model=NextQuestionResponse)
async def submit_answer(request: AnswerRequest) -> NextQuestionResponse:
    try:
        # Allow answers to be submitted even if the question text hasn't been
        # fully generated/saved on the server yet. Verify session exists first.
        session = questionnaireService.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="會話不存在或問題不存在")
        current_question = questionnaireService.get_current_question(
            request.session_id
        ) or ""
        sentiment_scores, stress_scores = (
            analysisService.analyze_user_response(
                request.answer,
                current_question,
            )
        )
        success = await questionnaireService.save_response(
            request.session_id, request.answer, sentiment_scores, stress_scores
        )
        if not success:
            raise HTTPException(status_code=400, detail="儲存回答失敗")
        if questionnaireService.is_questionnaire_complete(request.session_id):
            advice, analysis, profile, investor_type = await _finalize_session(
                request.session_id
            )
            return NextQuestionResponse(
                has_next_question=False,
                advice=advice,
                profile=profile,
                investor_type=investor_type,
                analysis=analysis if isinstance(analysis, dict) else None,
            )
        progress = questionnaireService.get_progress(request.session_id)
        all_responses = questionnaireService.get_all_responses(
            request.session_id
        )
        next_question, next_meta = await _generate_and_save_next(
            request.session_id, progress["current"] + 1, all_responses
        )
        return NextQuestionResponse(
            has_next_question=True,
            question=next_question,
            question_number=progress["current"] + 1,
            total_questions=progress["total"],
            question_meta=(next_meta if next_meta else None),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("提交答案時發生錯誤: %s", e)
        raise HTTPException(status_code=500, detail="伺服器內部錯誤")


@router.post("/stream-question")
async def stream_question(request: StreamQuestionRequest):
    try:
        # Ensure session exists; streaming is allowed even if the question
        # for the target number has not been pre-generated yet.
        session = questionnaireService.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="會話不存在或問題不存在")
        progress = questionnaireService.get_progress(request.session_id)
        all_responses = questionnaireService.get_all_responses(
            request.session_id
        )
        if not ollamaService.is_api_available():
            logger.error("AI model unavailable for streaming request")
            raise HTTPException(
                status_code=503,
                detail=(
                    "AI model not available for streaming generation"
                ),
            )
        target_number = request.question_number or (progress["current"] + 1)
        
        async def generate_stream():
            async for chunk in ollamaService.stream_question_generation(
                target_number, TOTAL_QUESTIONS, all_responses
            ):
                if chunk.get("done") and chunk.get("question"):
                    questionnaireService.save_generated_question(
                        request.session_id,
                        chunk["question"],
                        index=target_number - 1,
                        meta=chunk.get("meta") if chunk.get("meta") else None,
                    )
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("串流問題時發生錯誤: %s", e)
        raise HTTPException(status_code=500, detail="伺服器內部錯誤")


@router.post("/save-question")
async def save_question(request: SaveQuestionRequest) -> Dict[str, Any]:
    try:
        session = questionnaireService.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="會話不存在或問題不存在")
        current_question = questionnaireService.get_current_question(
            request.session_id
        ) or ""
        sentiment_scores, stress_scores = (
            analysisService.analyze_user_response(
                request.answer,
                current_question,
            )
        )
        success = await questionnaireService.save_response(
            request.session_id, request.answer, sentiment_scores, stress_scores
        )
        if not success:
            raise HTTPException(status_code=400, detail="儲存回答失敗")
        if questionnaireService.is_questionnaire_complete(request.session_id):
            advice, analysis, profile, investor_type = await _finalize_session(
                request.session_id
            )
            return {
                "success": True,
                "is_complete": True,
                "advice": advice,
                "profile": profile,
                "investor_type": investor_type,
                "analysis": analysis if isinstance(analysis, dict) else None,
            }
        progress = questionnaireService.get_progress(request.session_id)
        return {
            "success": True,
            "is_complete": False,
            "next_question_number": progress["current"] + 1,
            "total_questions": progress["total"],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("儲存問題時發生錯誤: %s", e)
        raise HTTPException(status_code=500, detail="伺服器內部錯誤")
