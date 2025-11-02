from typing import Dict, List
from models import sentimentModel  # 移除 stressModel
from config import ENABLE_CONTEXT_ANALYSIS


class AnalysisService:
    def __init__(self):
        pass

    def sanitize_sentiment_output(self, raw) -> Dict[str, float]:
        """解析 SentimentModel 輸出，提取 negative、neutral、positive 分數"""
        result = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
        try:
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

    def analyze_user_response(self, text: str, question: str = "") -> (
            tuple[Dict[str, float], Dict[str, float]]
            ):
        """分析使用者回應，回傳情緒與（空的）壓力分數以維持相容 API"""
        if ENABLE_CONTEXT_ANALYSIS and question.strip():
            analysis_text = f"問題：{question.strip()} 回答：{text.strip()}"
            print(f"📊 分析上下文: {analysis_text[:100]}...")
        else:
            analysis_text = text.strip()
            if question:
                print(f"⚠️ 有問題但未使用上下文分析: {question[:50]}...")
            print(f"📊 分析回答: {analysis_text[:50]}...")

        # 只執行情緒分析（stressModel 已移除）
        sentiment_raw = sentimentModel.analyze(analysis_text)

        sentiment_scores = self.sanitize_sentiment_output(sentiment_raw)
        stress_scores = {}  # 回傳空 dict 以保持呼叫端相容性

        print(f"🎭 情緒分析結果: {sentiment_scores}")
        # 移除壓力分析輸出

        return sentiment_scores, stress_scores

    # 新增：由整個回應列表計算 profile（五項指標）
    def compute_profile(self, all_responses: List[Dict]) -> Dict[str, int]:
        """
        all_responses 為 questionnaire_service 存的 response_data 列表：
        每項通常包含 keys: question, answer, sentiment, stress
        回傳 risk, stability, confidence, patience, sensitivity（0-100）
        """
        risk = 50
        stability = 50
        confidence = 50
        patience = 50
        sensitivity = 50

        for r in all_responses:
            ans = (r.get("answer") or "").strip()
            # 先嘗試從 answer 抽出 Likert 數值（開頭數字或 "N — ..." 格式）
            likert_val = None
            try:
                # 若格式為 "5 — 描述" 或 "5-描述"
                if ans and (ans[0].isdigit()):
                    # 取首個數字
                    likert_val = int(ans[0])
                    if likert_val < 1 or likert_val > 5:
                        likert_val = None
            except Exception:
                likert_val = None

            if likert_val is not None:
                v = likert_val
                risk += (v - 3) * 8
                stability += (3 - v) * 6
                confidence += (v - 3) * 6
                patience += (v - 3) * 4
                sensitivity += (3 - v) * 6
                continue

            # 非 Likert：以文字關鍵字映射
            text = ans.lower()
            if any(k in text for k in ["加碼", "買入", "進場", "冒險", "高風險"]):
                risk += 12
                confidence += 8
                sensitivity += 6
            elif any(k in text for k in ["賣出", "逃離", "恐慌", "立刻賣出", "減碼"]):
                risk -= 12
                stability -= 8
                sensitivity += 10
            elif any(k in text for k in ["觀望", "冷靜", "等待", "持有", "保守"]):
                stability += 10
                patience += 8
                risk -= 4
            else:
                # 長文字視為較高參與與信心
                if len(text) > 80:
                    confidence += 6
                    patience += 4

        # clamp 0..100
        def clamp(x): return max(0, min(100, round(x)))
        return {
            "risk": clamp(risk),
            "stability": clamp(stability),
            "confidence": clamp(confidence),
            "patience": clamp(patience),
            "sensitivity": clamp(sensitivity),
        }

    # 新增：依 profile 決定投資者類型
    def classify_investor(self, profile: Dict[str, int]) -> str:
        p = profile
        if p["risk"] > 60 and p["stability"] < 40:
            return "波動型（情緒受市場影響）"
        if p["risk"] > 60 and p["stability"] >= 40:
            return "探險型（高風險偏好）"
        if p["risk"] <= 40 and p["stability"] >= 60:
            return "冷靜型（理性決策）"
        if p["risk"] <= 40 and p["stability"] < 60:
            return "謹慎型（保守穩健）"
        return "綜合型（中庸平衡）"
