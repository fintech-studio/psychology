from typing import Dict, List, Optional
import uuid
import threading
from config import TOTAL_QUESTIONS

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
    
    def save_generated_question(self, session_id: str, question: str) -> bool:
        """儲存動態生成的問題"""
        with self.sessions_lock:
            session = self.sessions.get(session_id)
            if not session:
                return False
            
            session["questions"].append(question)
            return True
    
    def save_response(self, session_id: str, answer: str, sentiment_scores: Dict[str, float], stress_scores: Dict[str, float]) -> bool:
        """儲存回答"""
        with self.sessions_lock:
            session = self.sessions.get(session_id)
            if not session:
                return False
            
            current_index = session["current_question"]
            questions = session.get("questions", [])
            
            if current_index >= len(questions):
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