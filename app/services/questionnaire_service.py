from typing import Dict, List, Optional
import uuid
import threading
import logging
import re
from config import TOTAL_QUESTIONS

logger = logging.getLogger(__name__)


class QuestionnaireService:
    def __init__(self):
        # 會話管理
        self.sessions: Dict[str, Dict] = {}
        self.sessions_lock = threading.Lock()

        # 問題設定
        self.total_questions = TOTAL_QUESTIONS

    def create_session(self) -> str:
        """建立新的會話"""
        session_id = str(uuid.uuid4())
        with self.sessions_lock:
            self.sessions[session_id] = {
                "current_question": 0,
                "responses": [],
                "questions": []  # 儲存動態生成的問題
            }
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict]:
        """取得會話資料"""
        with self.sessions_lock:
            return self.sessions.get(session_id)

    def get_current_question(self, session_id: str) -> Optional[str]:
        """取得當前問題（如果已生成）"""
        session = self.get_session(session_id)
        if not session:
            return None

        current_index = session["current_question"]
        questions = session.get("questions", [])

        # 如果問題已經生成，返回它
        if current_index < len(questions):
            return questions[current_index]

        # 如果問題還沒生成，返回 None（需要動態生成）
        return None

    def save_generated_question(
        self,
        session_id: str,
        question: str,
        index: Optional[int] = None,
        meta: Optional[Dict] = None,
    ) -> bool:
        """儲存動態生成的問題"""
        with self.sessions_lock:
            session = self.sessions.get(session_id)
            if not session:
                return False

            current_index = session["current_question"]
            if index is not None and isinstance(index, int) and index >= 0:
                current_index = index
            questions = session["questions"]

            # 確保 questions 列表足夠長，填充空位置
            while len(questions) <= current_index:
                questions.append("")

            # 在正確的索引位置儲存問題
            questions[current_index] = question
            # 儲存結構化 meta（若模型提供）
            if meta is not None:
                if "questions_meta" not in session:
                    session["questions_meta"] = []
                metas = session["questions_meta"]
                while len(metas) <= current_index:
                    metas.append({})
                metas[current_index] = meta
            # If question contains placeholder tokens like '選擇A' or
            # single-letter options, sanitize by removing placeholders.
            try:
                if (
                    re.search(r"\b(選擇|選項)\s*[A-D]\b", question)
                    or re.search(r"\b[A-D]\b\s*/\s*\b[A-D]\b", question)
                ):
                    # remove tokens like '選擇A', '選項A' and single letters
                    sanitized = re.sub(r"(?:選擇|選項)?\s*[A-D]", "", question)
                    sanitized = re.sub(r"\b[A-D]\b", "", sanitized)
                    sanitized = re.sub(r"\s*/\s*", " / ", sanitized).strip()
                    # store sanitized question (no defaults injected)
                    questions[current_index] = sanitized
                    logger.debug(
                        "Sanitized placeholder question: %s -> %s",
                        question,
                        sanitized,
                    )
            except Exception:
                logger.exception(
                    "Failed to sanitize placeholder options for question: %s",
                    question,
                )

            return True

    def save_response(self, session_id: str, answer: str,
                      sentiment_scores: Dict[str, float],
                      stress_scores: Dict[str, float]) -> bool:
        """儲存回答"""
        with self.sessions_lock:
            session = self.sessions.get(session_id)
            if not session:
                return False

            current_index = session["current_question"]
            questions = session.get("questions", [])

            if current_index >= len(questions) or not questions[current_index]:
                logger.warning(
                    "⚠️ 警告：第 %s 題問題尚未正確儲存",
                    current_index + 1,
                )
                return False

            # 儲存回答和分析結果
            response_data = {
                "question": questions[current_index],
                "answer": answer,
                "sentiment": sentiment_scores,
                "stress": stress_scores
            }
            session["responses"].append(response_data)

            # 移動到下一個問題
            session["current_question"] += 1

            return True

    def is_questionnaire_complete(self, session_id: str) -> bool:
        """檢查問卷是否完成"""
        session = self.get_session(session_id)
        if not session:
            return False
        return session["current_question"] >= self.total_questions

    def get_all_responses(self, session_id: str) -> List[Dict]:
        """取得所有回答"""
        session = self.get_session(session_id)
        if not session:
            return []
        return session["responses"]

    def get_progress(self, session_id: str) -> Dict[str, int]:
        """取得進度資訊"""
        session = self.get_session(session_id)
        if not session:
            return {"current": 0, "total": self.total_questions}
        return {
            "current": session["current_question"],
            "total": self.total_questions
        }

    def delete_session(self, session_id: str) -> bool:
        """刪除會話"""
        with self.sessions_lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                return True
            return False
