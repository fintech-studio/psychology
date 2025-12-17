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
            # 若先前已存在以佔位字串儲存的回答，回填該回答中的 question 欄位
            try:
                responses = session.get("responses", [])
                if (
                    isinstance(responses, list)
                    and len(responses) > current_index
                    and responses[current_index].get("question") in (
                        None, "", "(問題尚未生成)")
                ):
                    responses[current_index]["question"] = question
                    logger.info(
                        "回填回答 %s 的 question 欄位：%s",
                        current_index,
                        session["responses"][current_index]["question"],
                    )
            except Exception:
                logger.exception("回填已儲存回答的題目時發生錯誤：%s", session_id)
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

    def _ensure_question_slot(self, session: Dict, index: int):
        """確保 questions 列表長度足以放置 index。"""
        questions = session.get("questions", [])
        while len(questions) <= index:
            questions.append("")
        session["questions"] = questions

    async def save_response(self, session_id: str, answer: str,
                            sentiment_scores: Dict[str, float],
                            stress_scores: Dict[str, float]) -> bool:
        """儲存回答。

        當對應問題尚未生成時，會嘗試立即以 Ollama 生成並回填題目；若生成失敗，則以佔位字串先行儲存，之後可由其他生成程序回填真實題目。
        """
        PLACEHOLDER = "(問題尚未生成)"
        with self.sessions_lock:
            session = self.sessions.get(session_id)
            if not session:
                return False

            current_index = session["current_question"]
            questions = session.get("questions", [])

            # 如果問題尚未生成，嘗試立即生成並回填
            if current_index >= len(questions) or not questions[current_index]:
                # try to generate synchronously if Ollama available
                try:
                    from services import ollamaService
                    if ollamaService.is_api_available():
                        # previous responses so far (to pass for context)
                        prev = session.get("responses", [])
                        # Ollama expects 1-based question number
                        qnum = current_index + 1
                        try:
                            qtext, qmeta = await (
                                ollamaService.generate_dynamic_question(
                                    current_number=qnum,
                                    total_questions=self.total_questions,
                                    previous_responses=prev,
                                )
                            )
                            # ensure slot exists and save generated question
                            self._ensure_question_slot(session, current_index)
                            session["questions"][current_index] = qtext
                            if qmeta is not None:
                                if "questions_meta" not in session:
                                    session["questions_meta"] = []
                                metas = session["questions_meta"]
                                while len(metas) <= current_index:
                                    metas.append({})
                                metas[current_index] = qmeta
                        except Exception:
                            logger.exception(
                                "Immediate generation for question %s failed",
                                qnum,
                            )
                            # fallback to placeholder below
                    else:
                        logger.warning(
                            "⚠️ 第 %s 題問題尚未生成，AI model unavailable;"
                            " saving placeholder",
                            current_index + 1,
                        )
                except Exception:
                    logger.exception(
                        "Error while attempting immediate question generation"
                    )

                # Ensure slot exists and write placeholder if still missing
                self._ensure_question_slot(session, current_index)
                if not session["questions"][current_index]:
                    session["questions"][current_index] = PLACEHOLDER

            # 儲存回答和分析結果
            response_data = {
                "question": session["questions"][current_index],
                "answer": answer,
                "sentiment": sentiment_scores,
                "stress": stress_scores,
            }

            # Attach any available structured meta
            # (options, options_score, dimension)
            try:
                metas = session.get("questions_meta", [])
                if len(metas) > current_index and isinstance(
                        metas[current_index], dict):
                    qmeta = metas[current_index]
                    response_data["meta"] = qmeta
                    # If options present, attempt to
                    # resolve selected option index and score
                    opts = qmeta.get("options") if isinstance(
                        qmeta.get("options"), list) else None
                    scores = qmeta.get("options_score") if isinstance(
                        qmeta.get("options_score"), list) else None
                    sel_idx = None
                    sel_score = None
                    if opts:
                        # Try exact match / case-insensitive match
                        ans_norm = (answer or "").strip()
                        for i, o in enumerate(opts):
                            if ans_norm == o or ans_norm.lower() == o.lower():
                                sel_idx = i
                                break
                        # Try letter-based match (A/B/C)
                        if sel_idx is None:
                            m = re.search(r"\b([A-D])\b", ans_norm, re.I)
                            if m:
                                li = ord(m.group(1).upper()) - ord('A')
                                if 0 <= li < len(opts):
                                    sel_idx = li
                        # Fallback: try partial containment
                        if sel_idx is None:
                            for i, o in enumerate(opts):
                                if o.lower() in ans_norm.lower() or (
                                        ans_norm.lower() in o.lower()):
                                    sel_idx = i
                                    break
                        if sel_idx is not None and scores and (
                                0 <= sel_idx < len(scores)):
                            try:
                                sel_score = float(scores[sel_idx])
                                # Clamp 0..1
                                sel_score = max(0.0, min(1.0, sel_score))
                            except Exception:
                                sel_score = None
                        # attach found values even if None for clarity
                        response_data["selected_option_index"] = sel_idx
                        response_data["selected_option_score"] = sel_score
            except Exception:
                logger.exception(
                    "Failed to attach question meta to response: %s",
                    session_id)

            session["responses"].append(response_data)

            # 移動到下一個問題
            session["current_question"] += 1

            return True

    # Note: previous implementation of save_response that returned False when
    # the current question wasn't present has been removed in favor of the
    # placeholder-capable implementation above.

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
