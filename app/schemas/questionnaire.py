from pydantic import BaseModel
from typing import Optional

class StartResponse(BaseModel):
    session_id: str

class AnswerRequest(BaseModel):
    session_id: str
    answer: str

class NextQuestionResponse(BaseModel):
    has_next_question: bool = True
    question: Optional[str] = None
    question_number: Optional[int] = None
    total_questions: Optional[int] = None
    advice: Optional[str] = None

class StreamQuestionRequest(BaseModel):
    session_id: str

class SaveQuestionRequest(BaseModel):
    session_id: str
    question: str
    answer: str