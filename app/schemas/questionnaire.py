from pydantic import BaseModel
from typing import Optional, Dict


class StartResponse(BaseModel):
    session_id: str
    question: Optional[str] = None
    question_number: Optional[int] = None
    total_questions: Optional[int] = None


class AnswerRequest(BaseModel):
    session_id: str
    answer: str


class NextQuestionResponse(BaseModel):
    has_next_question: bool = True
    question: Optional[str] = None
    question_number: Optional[int] = None
    total_questions: Optional[int] = None
    advice: Optional[str] = None
    # 新增：完成時後端回傳的分析檔案與類型
    profile: Optional[Dict[str, int]] = None
    investor_type: Optional[str] = None


class StreamQuestionRequest(BaseModel):
    session_id: str
    question_number: Optional[int] = None


class SaveQuestionRequest(BaseModel):
    session_id: str
    question: str
    answer: str
