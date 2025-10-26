from typing import Dict
from models.SentimentModel import SentimentModel
from models.StressModel import StressModel
from config import ENABLE_CONTEXT_ANALYSIS

class AnalysisService:
    def __init__(self):
        self.sentiment_model = None
        self.stress_model = None
    
    def ensure_models_loaded(self):
        """確保模型已載入"""
        if self.sentiment_model is None:
            self.sentiment_model = SentimentModel()
        if self.stress_model is None:
            self.stress_model = StressModel()
    
    def sanitize_sentiment_output(self, raw) -> Dict[str, float]:
        """解析 SentimentModel 輸出，提取 negative、neutral、positive 分數"""
        result = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
        
        try:
            # 處理巢狀列表格式: [[{...}]]
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

    def sanitize_stress_output(self, raw) -> Dict[str, float]:
        """解析 StressModel 輸出，提取 stress、not_stress 分數"""
        result = {"stress": 0.0, "not_stress": 0.0}
        
        try:
            # 處理巢狀列表格式: [[{...}]]
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
                
                if "stressed" in label and "not" not in label:
                    result["stress"] = score
                elif "not stressed" in label or "not" in label:
                    result["not_stress"] = score
                    
        except Exception as e:
            print(f"解析壓力輸出時發生錯誤: {e}")
            
        return result
    
    def analyze_user_response(self, text: str, question: str = "") -> tuple[Dict[str, float], Dict[str, float]]:
        """分析使用者回應，回傳情緒和壓力分數"""
        self.ensure_models_loaded()
        
        # 根據配置決定是否使用上下文分析
        if ENABLE_CONTEXT_ANALYSIS and question.strip():
            # 組合問題和回答提供完整上下文
            # 格式: "問題：{question} 回答：{text}" 
            analysis_text = f"問題：{question.strip()} 回答：{text.strip()}"
            print(f"📊 分析上下文: {analysis_text[:100]}...")  # 顯示前100字符用於調試
        else:
            # 如果沒有問題或未啟用上下文分析，就直接用回答
            analysis_text = text.strip()
            print(f"📊 分析回答: {analysis_text[:50]}...")  # 顯示前50字符用於調試
        
        # 執行分析（使用包含上下文的文本）
        sentiment_raw = self.sentiment_model.analyze(analysis_text)
        stress_raw = self.stress_model.analyze(analysis_text)
        
        # 解析結果
        sentiment_scores = self.sanitize_sentiment_output(sentiment_raw)
        stress_scores = self.sanitize_stress_output(stress_raw)
        
        # 輸出分析結果用於調試
        print(f"🎭 情緒分析結果: {sentiment_scores}")
        print(f"😰 壓力分析結果: {stress_scores}")
        
        return sentiment_scores, stress_scores