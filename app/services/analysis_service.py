from typing import Dict, List
import logging
# deferred import of models done in __init__
from config import ENABLE_CONTEXT_ANALYSIS

logger = logging.getLogger(__name__)

# Profile scoring weights and defaults
DEFAULT_PROFILE_SCORE = 50
KEYWORD_WEIGHTS = {
    "buy": {"risk": 12, "confidence": 8, "sensitivity": 6},
    "sell": {"risk": -12, "stability": -8, "sensitivity": 10},
    "hold": {"stability": 10, "patience": 8, "risk": -4},
    "text_len": {"confidence": 6, "patience": 4},
}


class AnalysisService:
    def __init__(self):
        # ensure sentiment model is initialized (deferred load)
        try:
            from models import init_models
            init_models()
            from models import sentimentModel as sm
            self.sentiment = sm
        except Exception as e:
            logger.exception("SentimentModel init failed: %s", e)
            self.sentiment = None

    def sanitize_sentiment_output(self, raw) -> Dict[str, float]:
        """解析 SentimentModel 輸出，提取 negative、neutral、positive 分數"""
        result = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
        try:
            data = None
            if isinstance(raw, dict):
                # case: {label: score} or {'labels': []}
                if "labels" in raw and isinstance(raw["labels"], list):
                    data = raw["labels"]
                else:
                    # map-like
                    for label, score in raw.items():
                        try:
                            s = float(score)
                        except Exception:
                            continue
                        key_lower = label.lower()
                        if "neg" in key_lower:
                            result["negative"] = s
                        elif "pos" in key_lower:
                            result["positive"] = s
                        elif "neu" in key_lower:
                            result["neutral"] = s
                    data = None
            elif isinstance(raw, list) and raw and isinstance(raw[0], list):
                data = raw[0]
            elif isinstance(raw, list):
                data = raw
            else:
                return result

            if data:
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    label = item.get("label", "").lower()
                    try:
                        score = float(item.get("score", 0.0))
                    except Exception:
                        score = 0.0
                    if "negative" in label or "neg" in label:
                        result["negative"] = score
                    elif "positive" in label or "pos" in label:
                        result["positive"] = score
                    elif "neutral" in label or "neu" in label:
                        result["neutral"] = score
        except Exception as e:
            logger.exception("解析情緒輸出時發生錯誤: %s", e)

        # normalize to sum 1.0 if possible
        try:
            total = result["negative"] + result["neutral"] + result["positive"]
            if total > 0:
                result["negative"] = result["negative"] / total
                result["neutral"] = result["neutral"] / total
                result["positive"] = result["positive"] / total
        except Exception:
            # ignore normalization errors
            pass

        return result

    def analyze_user_response(self, text: str, question: str = "") -> (
        tuple[Dict[str, float], Dict[str, float]]
    ):
        """分析使用者回應，回傳情緒與（空的）壓力分數以維持相容 API"""
        if ENABLE_CONTEXT_ANALYSIS and question.strip():
            analysis_text = f"問題：{question.strip()} 回答：{text.strip()}"
            logger.debug("分析上下文: %s", analysis_text[:100])
        else:
            analysis_text = text.strip()
            if question:
                logger.warning("有問題但未使用上下文分析: %s", question[:50])
            logger.debug("分析回答: %s", analysis_text[:50])

        # 只執行情緒分析（stressModel 已移除）
        sentiment_raw = None
        if self.sentiment:
            try:
                sentiment_raw = self.sentiment.analyze(analysis_text)
            except Exception as e:
                logger.exception("Sentiment analyze failed: %s", e)
                sentiment_raw = None

        sentiment_scores = self.sanitize_sentiment_output(sentiment_raw)
        stress_scores = {}  # 回傳空 dict 以保持呼叫端相容性

        logger.debug("情緒分析結果: %s", sentiment_scores)
        # 移除壓力分析輸出

        return sentiment_scores, stress_scores

    # 新增：由整個回應列表計算 profile（五項指標）
    def compute_profile(self, all_responses: List[Dict]) -> Dict[str, int]:
        """
        all_responses 為 questionnaire_service 存的 response_data 列表：
        每項通常包含 keys: question, answer, sentiment, stress
        回傳 risk, stability, confidence, patience, sensitivity（0-100）
        """
        risk = DEFAULT_PROFILE_SCORE
        stability = DEFAULT_PROFILE_SCORE
        confidence = DEFAULT_PROFILE_SCORE
        patience = DEFAULT_PROFILE_SCORE
        sensitivity = DEFAULT_PROFILE_SCORE

        for r in all_responses:
            ans = (r.get("answer") or "").strip()
            # Non-numeric answers: map keywords and text length to scores

            # 以文字關鍵字與長度映射
            text = ans.lower()
            if any(k in text for k in ["加碼", "買入", "進場", "冒險", "高風險"]):
                risk += KEYWORD_WEIGHTS["buy"]["risk"]
                confidence += KEYWORD_WEIGHTS["buy"]["confidence"]
                sensitivity += KEYWORD_WEIGHTS["buy"]["sensitivity"]
            elif any(k in text for k in ["賣出", "逃離", "恐慌", "立刻賣出", "減碼"]):
                risk += KEYWORD_WEIGHTS["sell"]["risk"]
                stability += KEYWORD_WEIGHTS["sell"]["stability"]
                sensitivity += KEYWORD_WEIGHTS["sell"]["sensitivity"]
            elif any(k in text for k in ["觀望", "冷靜", "等待", "持有", "保守"]):
                stability += KEYWORD_WEIGHTS["hold"]["stability"]
                patience += KEYWORD_WEIGHTS["hold"]["patience"]
                risk += KEYWORD_WEIGHTS["hold"]["risk"]
            else:
                # 長文字視為較高參與與信心
                if len(text) > 80:
                    confidence += KEYWORD_WEIGHTS["text_len"]["confidence"]
                    patience += KEYWORD_WEIGHTS["text_len"]["patience"]

        # clamp 0..100
        def clamp(x: float) -> int:
            return max(0, min(100, round(x)))
        profile = {
            "risk": clamp(risk),
            "stability": clamp(stability),
            "confidence": clamp(confidence),
            "patience": clamp(patience),
            "sensitivity": clamp(sensitivity),
        }
        logger.debug("Computed user profile: %s", profile)
        return profile
        # log computed profile

    # 新增：依 profile 決定投資者類型
    def classify_investor(self, profile: Dict[str, int]) -> str:
        p = profile
        logger.debug("Classifying investor for profile: %s", profile)
        if p["risk"] > 60 and p["stability"] < 40:
            return "波動型（情緒受市場影響）"
        if p["risk"] > 60 and p["stability"] >= 40:
            return "探險型（高風險偏好）"
        if p["risk"] <= 40 and p["stability"] >= 60:
            return "冷靜型（理性決策）"
        if p["risk"] <= 40 and p["stability"] < 60:
            return "謹慎型（保守穩健）"
        return "綜合型（中庸平衡）"

    def describe_investor(self, profile: Dict[str, int]) -> str:
        """Return a short textual description based on investor profile."""
        investor_type = self.classify_investor(profile)
        if investor_type.startswith("波動型"):
            return (
                "您可能對短期價格波動較敏感，情緒波動會影響決策，需加強風險管理與情緒調節。"
            )
        if investor_type.startswith("探險型"):
            return (
                "您偏好高風險投資與高報酬，適合在資產中留有一部分承擔風險，並搭配風險分散策略。"
            )
        if investor_type.startswith("冷靜型"):
            return (
                "您在市場波動中能保持冷靜，較偏向理性決策，可利用長期策略提升投資報酬。"
            )
        if investor_type.startswith("謹慎型"):
            return (
                "您偏好保守穩健的投資，較少被情緒影響，適合以收益穩健、風險可控的配置為主。"
            )
        return (
            "您的風險與穩定性影響衡平，建議依個人目標與時間視窗進一步制定投資策略。"
        )

    def compute_time_horizon(self, all_responses: List[Dict]) -> int:
        """Estimate time-horizon preference from text answers
        (0..100, where 100 means very long-term)."""
        score = 50
        for r in all_responses:
            q = (r.get("question") or "").lower()
            a = (r.get("answer") or "").lower()
            if any(k in q for k in ["長期", "長線", "退休", "長期投資"]):
                score += 20
            if any(k in a for k in ["長期", "長線", "退休", "長期投資"]):
                score += 20
            if any(k in q for k in ["短期", "短線", "短期獲利"]):
                score -= 20
            if any(k in a for k in ["短期", "短線", "短期獲利"]):
                score -= 20
        return max(0, min(100, round(score)))

    def compute_stress_index(self, all_responses: List[Dict]) -> int:
        """Compute a psychological stress index (0..100) combining
        textual answers, sentiment negative proportion,
        and stress-related keyword counts.
        """
        base = 40
        negative_total = 0.0
        count = 0
        keyword_hits = 0
        stress_keywords = ["壓力", "睡眠", "焦慮", "失眠", "緊張", "心悸", "睡不好"]

        for r in all_responses:
            ans = (r.get("answer") or "")

            # sentiment negative
            sentiment = r.get("sentiment", {})
            negative_total += (sentiment.get("negative", 0.0) or 0.0)
            count += 1

            # keyword hits in answer
            la = ans.lower()
            for k in stress_keywords:
                if k in la:
                    keyword_hits += 1

        # negative sentiment contribution:
        # avg_negative in 0..1 -> scale to 0..40
        if count > 0:
            avg_negative = negative_total / count
            base += avg_negative * 40.0

        # keyword hits each add +3
        base += keyword_hits * 3.0

        # clamp 0..100
        return max(0, min(100, round(base)))

    def compute_radar(self, profile: Dict[str, int],
                      stress_index: int, time_horizon: int) -> Dict[str, int]:
        """Return radar-ready dict with tuned dimensions.
        Dimensions: risk, stability, confidence, patience
        (inverse->impulsivity), sensitivity,
        time_horizon, stress
        """
        radar = {
            "risk": max(0, min(100, int(profile.get("risk", 50)))),
            "stability": max(0, min(100, int(profile.get("stability", 50)))),
            "confidence": max(0, min(100, int(profile.get("confidence", 50)))),
            "impulsivity": max(0, min(100, 100 - int(profile.get(
                "patience", 50)))),
            "sensitivity": max(0, min(100, int(profile.get(
                "sensitivity", 50)))),
            "time_horizon": max(0, min(100, int(time_horizon))),
            "stress": max(0, min(100, int(stress_index))),
        }
        return radar
