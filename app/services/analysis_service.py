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

        # Redesigned behavior:
        # - Use structured `meta` and `options_score` when available.
        # - Prefer `selected_option_score` when present;
        #   otherwise derive a selection
        #   from `selected_option_index`,
        #   answer text, or fallback to the highest score.
        # - If no structured info available,
        #   use sentiment as a light-weight fallback.
        # - Apply dimension-specific scales to map selection strength (0..1)
        #   into profile attribute deltas.

        DIMENSION_SCALES = {
            # dimension: {profile_attr: scale}
            "risk": {"risk": 36.0, "confidence": 12.0, "sensitivity": 8.0},
            "stress": {
                "stability": -36.0, "patience": -18.0, "sensitivity": 12.0},
            "emotion": {
                "sensitivity": 24.0, "confidence": 10.0, "stability": -6.0},
            "time": {"patience": 36.0, "risk": -12.0, "confidence": 6.0},
            "decision": {
                "risk": 24.0, "confidence": 18.0, "sensitivity": 12.0},
        }

        def _determine_selected_score(r: Dict) -> float | None:
            """Determine a selected score (0..1) for the response r.
            Returns None if no information can be derived."""
            # direct selected score first
            sel = r.get("selected_option_score")
            if sel is not None:
                try:
                    return float(sel)
                except Exception:
                    pass

            # try selected index + meta options_score
            idx = r.get("selected_option_index")
            meta = r.get("meta") or {}
            opts_scores = None
            if isinstance(meta.get("options_score"), list):
                try:
                    opts_scores = [float(x) for x in meta.get("options_score")]
                except Exception:
                    opts_scores = None
            if idx is not None and isinstance(idx, int) and opts_scores:
                if 0 <= idx < len(opts_scores):
                    return float(opts_scores[idx])

            # try to match answer to options (best-effort)
            ans = (r.get("answer") or "").strip()
            opts = meta.get("options") if isinstance(meta.get("options"),
                                                     list) else None
            if opts and opts_scores:
                # exact / case-insensitive match
                for i, o in enumerate(opts):
                    if ans == o or ans.lower() == o.lower():
                        try:
                            return float(opts_scores[i])
                        except Exception:
                            return None
                # partial containment fallback
                for i, o in enumerate(opts):
                    if o.lower() in ans.lower() or ans.lower() in o.lower():
                        try:
                            return float(opts_scores[i])
                        except Exception:
                            return None
                # fallback to strongest option if user didn't select clearly
                try:
                    return float(max(opts_scores))
                except Exception:
                    return None

            # As last resort use sentiment-derived soft signal
            sentiment = r.get("sentiment", {})
            pos = float(sentiment.get("positive", 0.0) or 0.0)
            neg = float(sentiment.get("negative", 0.0) or 0.0)
            net = pos - neg
            # map net (-1..1) -> roughly (0..1) centered at 0.5
            try:
                return max(0.0, min(1.0, 0.5 + net * 0.25))
            except Exception:
                return None

        # Aggregate contributions
        contribution_counts = {
            "risk": 0,
            "stability": 0,
            "confidence": 0,
            "patience": 0,
            "sensitivity": 0}

        for r in all_responses:
            meta = r.get("meta") or {}
            dim = (meta.get("dimension") or "").lower()
            sel_score = _determine_selected_score(r)
            if sel_score is None:
                # nothing we can do for this response
                continue
            scales = DIMENSION_SCALES.get(dim)
            if not scales:
                # unknown dimension: skip but count as ignored
                continue

            for attr, scale in scales.items():
                delta = (float(sel_score) - 0.5) * float(scale)
                if attr == "risk":
                    risk += delta
                    contribution_counts["risk"] += 1
                elif attr == "stability":
                    stability += delta
                    contribution_counts["stability"] += 1
                elif attr == "confidence":
                    confidence += delta
                    contribution_counts["confidence"] += 1
                elif attr == "patience":
                    patience += delta
                    contribution_counts["patience"] += 1
                elif attr == "sensitivity":
                    sensitivity += delta
                    contribution_counts["sensitivity"] += 1

        # Optionally normalize by number of contributions per attribute
        # (to reduce bias from many questions)
        for attr in ["risk",
                     "stability",
                     "confidence",
                     "patience",
                     "sensitivity"]:
            cnt = contribution_counts.get(attr, 0)
            if cnt > 0:
                # gently average the accumulated delta
                # by number of contributions
                # We already added deltas directly;
                # dividing by cnt reduces magnitude from many small questions
                # Using a mild normalizer
                if attr == "risk":
                    risk = DEFAULT_PROFILE_SCORE + (
                        risk - DEFAULT_PROFILE_SCORE) / (1 + 0.25 * (cnt - 1))
                else:
                    # for others use similar normalization
                    val = locals()[attr]
                    new_val = DEFAULT_PROFILE_SCORE + (
                        val - DEFAULT_PROFILE_SCORE) / (1 + 0.25 * (cnt - 1))
                    if attr == "stability":
                        stability = new_val
                    elif attr == "confidence":
                        confidence = new_val
                    elif attr == "patience":
                        patience = new_val
                    elif attr == "sensitivity":
                        sensitivity = new_val

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
        """Estimate time-horizon preference using
        question meta `dimension=='time'`
        and the `selected_option_score` (0..1).
        Returns 0..100, where 100 means very long-term.

        If no structured time-related responses exist, default to 50.
        """
        score = 50.0
        count = 0
        for r in all_responses:
            sel = r.get("selected_option_score")
            meta = r.get("meta") or {}
            dim = (meta.get("dimension") or "").lower()
            if dim != "time" or sel is None:
                continue
            # sel in 0..1 -> contribution scaled to +/-30
            score += (float(sel) - 0.5) * 60.0
            count += 1
        if count == 0:
            return 50
        return max(0, min(100, round(score)))

    def compute_stress_index(self, all_responses: List[Dict]) -> int:
        """Compute a psychological stress index (0..100).

        New approach:
        - Use negative sentiment average (as before) + `selected_option_score`
          contributions from questions with `dimension=='stress'`.
        - Each stress-related selected score contributes up to +30.
        """
        base = 40.0
        negative_total = 0.0
        count = 0
        stress_contrib = 0.0

        for r in all_responses:
            # sentiment negative
            sentiment = r.get("sentiment", {})
            negative_total += (sentiment.get("negative", 0.0) or 0.0)
            count += 1

            sel = r.get("selected_option_score")
            meta = r.get("meta") or {}
            dim = (meta.get("dimension") or "").lower()
            if dim == "stress" and sel is not None:
                try:
                    stress_contrib += float(sel) * 30.0
                except Exception:
                    pass

        if count > 0:
            avg_negative = negative_total / count
            base += avg_negative * 40.0

        base += stress_contrib
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
