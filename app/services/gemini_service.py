import asyncio
import os
from typing import List, Dict
import google.generativeai as genai
from dotenv import load_dotenv
from config import (
    GEMINI_MODEL_NAME, 
    GEMINI_TEMPERATURE, 
    GEMINI_MAX_TOKENS,
    GEMINI_ADVICE_TEMPERATURE,
    GEMINI_ADVICE_MAX_TOKENS,
    STREAM_DELAY
)

# 載入環境變數
load_dotenv()

class GeminiService:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.model_name = GEMINI_MODEL_NAME

        # print(f"Debug: GOOGLE_API_KEY = {self.api_key[:10] + '...' if self.api_key else 'None'}")

        if self.api_key:
            genai.configure(api_key=self.api_key)
            print("✅ Gemini API 金鑰設定成功")
            print("模型:", self.model_name)
        else:
            print("警告：未設定 GOOGLE_API_KEY，將使用模擬回應")
    
    async def generate_dynamic_question(self, current_number: int, total_questions: int, previous_responses: List[Dict] = None) -> str:
        """動態生成問題內容"""
        if not self.api_key:
            # 沒有 API 金鑰時使用預設問題
            # 財經/投資領域的預設題目（當沒有 API 金鑰時使用）
            fallback_questions = [
                "請描述您目前最重要的理財目標（例如購屋、退休、子女教育等），以及期望達成的時間表。",
                "您目前的風險承受能力如何？面對投資虧損時，您的反應通常是什麼？",
                "請說明您目前的資產配置（現金、股票、債券、不動產等）或您希望的配置方向。",
                "您每月可用於投資或儲蓄的金額約為多少？有無固定自動儲蓄或投資習慣？",
                "在過去五年中，您是否有重大財務事件（如負債、意外支出、投資獲利/虧損）影響您的理財規劃？"
            ]
            return fallback_questions[current_number - 1] if current_number <= len(fallback_questions) else f"請描述您對第{current_number}個投資/理財面向的看法。"
        
        # 根據問題編號和前面的回答生成提示
        if current_number == 1:
            prompt = f"""
你是一位專業的理財顧問 / 投資顧問，現在要開始一個包含{total_questions}個問題的理財與投資評估問卷。
請生成第一個問題，這個問題應該：
1. 探索使用者目前最重要的財務目標與時間框架
2. 幫助建立後續問卷的背景資訊（例如負債、可投資金額、時間線）
3. 用清楚且中性的語調
4. 大約20-40字
5. 使用繁體中文
6. 設計成能讓回答者提供具體、數值化或描述性的回應

請只回傳問題內容，不要包含任何格式標記或說明。
            """
        elif current_number <= total_questions // 2:
            # 前半部分問題：深入探討
            context_info = ""
            if previous_responses:
                last_response = previous_responses[-1]
                sentiment = last_response.get("sentiment", {})
                stress = last_response.get("stress", {})
                context_info = f"根據前一個回答的分析：情緒傾向（負面: {sentiment.get('negative', 0):.2f}, 正面: {sentiment.get('positive', 0):.2f}），壓力水平: {stress.get('stress', 0):.2f}"
            
            prompt = f"""
你是一位專業的理財顧問 / 投資顧問，這是{total_questions}題問卷中的第{current_number}題。
{context_info}

請根據使用者之前的回答生成一個深入探討的問題，這個問題應該：
1. 深入了解使用者的投資行為、風險管理或資產配置決策
2. 建立在前面回答的基礎上，例如現有負債、可投資金額、或過去投資經驗
3. 幫助評估其理財知識與實際行為差距
4. 大約20-40字
5. 使用繁體中文
6. 鼓勵具體且可操作的回應（數字、頻率或舉例）

請只回傳問題內容，不要包含任何格式標記或說明。
            """
        else:
            # 後半部分問題：整合和前瞻
            summary_info = ""
            if previous_responses and len(previous_responses) >= 2:
                avg_negative = sum(r.get("sentiment", {}).get("negative", 0) for r in previous_responses) / len(previous_responses)
                avg_stress = sum(r.get("stress", {}).get("stress", 0) for r in previous_responses) / len(previous_responses)
                summary_info = f"根據前面的回答分析，平均負面情緒: {avg_negative:.2f}, 平均壓力水平: {avg_stress:.2f}"
            
            prompt = f"""
你是一位專業的理財顧問 / 投資顧問，這是{total_questions}題問卷中的第{current_number}題（接近結尾）。
{summary_info}

請生成一個整合性或前瞻性的問題，這個問題應該：
1. 探討長期財務目標與達成策略（如退休規劃、稅務、保險與資產傳承）
2. 評估需補強的理財技能或資源需求
3. 幫助使用者訂出下一步可執行的理財計畫
4. 大約20-40字
5. 使用繁體中文
6. 幫助總結與展望，能促使使用者做出具體行動

請只回傳問題內容，不要包含任何格式標記或說明。
            """
        
        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=GEMINI_TEMPERATURE,
                    max_output_tokens=GEMINI_MAX_TOKENS,
                )
            )
            
            if response.text:
                # 清理回應，移除多餘的格式
                question = response.text.strip()
                # 移除可能的引號或格式標記
                question = question.replace('"', '').replace("'", '').replace('*', '')
                return question
            else:
                # 如果生成失敗，使用預設問題
                fallback_questions = [
                    "請描述您目前最重要的理財目標（例如購屋、退休、子女教育等）。",
                    "您目前的風險承受能力如何？面對投資虧損時通常會如何處理？",
                    "您目前有無固定的儲蓄或投資計畫？其頻率與金額約為何？"
                ]
                return fallback_questions[current_number - 1] if current_number <= len(fallback_questions) else "請描述您目前的理財狀況與目標。"
                
        except Exception as e:
            print(f"動態問題生成錯誤: {e}")
            # 發生錯誤時使用預設問題
            fallback_questions = [
                "最近在工作或學習上，有哪些情況讓你感到最大的壓力？請詳細描述一下。",
                "當面對困難的時候，你通常會採用什麼方式來處理？你覺得這些方式有效嗎？",
                "在你的日常生活中，有哪些因素或情境會直接影響你的情緒和心理狀態？"
            ]
            return fallback_questions[current_number - 1] if current_number <= len(fallback_questions) else "請描述您目前的心理狀態。"

    async def stream_question_generation(self, current_number: int, total_questions: int, previous_responses: List[Dict] = None):
        """串流方式生成問題"""
        # 首先動態生成問題內容
        question_text = await self.generate_dynamic_question(current_number, total_questions, previous_responses)
        
        # 串流顯示問題
        for char in question_text:
            yield {"text": char, "done": False}
            await asyncio.sleep(STREAM_DELAY)  # 延遲時間
        
        # 發送完成信號
        yield {"text": "", "done": True, "question": question_text}
    
    async def generate_content(self, all_responses: List[Dict]) -> str:
        """生成最終建議"""
        if not self.api_key:
            return "根據您的回答，建議您：1) 建立規律的壓力管理習慣 2) 尋求適當的社會支持 3) 學習正向的情緒調節技巧 4) 保持健康的生活作息"
        
        # 構建分析摘要和計算平均分數
        summary_lines = []
        total_negative = 0
        total_neutral = 0
        total_positive = 0
        total_stress = 0
        total_not_stress = 0
        response_count = len(all_responses)
        
        for i, response in enumerate(all_responses, 1):
            sentiment = response.get("sentiment", {})
            stress = response.get("stress", {})
            question = response.get("question", f"問題{i}")
            answer = response.get("answer", "無回答")
            
            # 累加分數用於計算平均值
            total_negative += sentiment.get('negative', 0)
            total_neutral += sentiment.get('neutral', 0)
            total_positive += sentiment.get('positive', 0)
            total_stress += stress.get('stress', 0)
            total_not_stress += stress.get('not_stress', 0)
            
            summary_lines.append(
                f"問題{i}: {question}\n"
                f"回答: {answer}\n"
                f"分析結果 - 負面情緒:{sentiment.get('negative', 0):.3f}, "
                f"正面情緒:{sentiment.get('positive', 0):.3f}, "
                f"壓力水平:{stress.get('stress', 0):.3f}\n"
            )
        
        # 計算平均分數
        if response_count > 0:
            avg_negative = total_negative / response_count
            avg_neutral = total_neutral / response_count
            avg_positive = total_positive / response_count
            avg_stress = total_stress / response_count
            avg_not_stress = total_not_stress / response_count
        else:
            avg_negative = avg_neutral = avg_positive = avg_stress = avg_not_stress = 0
        
        # 生成 prompt 給 Gemini
        prompt = f"""
請根據以下使用者在心理問卷中的情緒與壓力分析結果，提供個人化的心理健康建議：

整體平均分析結果：
- 平均負面情緒: {avg_negative:.3f}
- 平均中性情緒: {avg_neutral:.3f}
- 平均正面情緒: {avg_positive:.3f}
- 平均壓力水平: {avg_stress:.3f}
- 平均無壓力水平: {avg_not_stress:.3f}

詳細問答與分析：
{chr(10).join(summary_lines)}

請提供：
1. 心理狀態整體分析（基於平均分數）
2. 壓力管理建議
3. 情緒調適技巧
4. 具體的改善方案

請用繁體中文回答，內容要實用且易於執行。
注意：請不要使用任何 Markdown 格式標記，回答內容應該是純文字格式。
        """
        
        # Debug: 印出傳給 Gemini 的完整內容
        # print(f"Debug - 回答數量: {response_count}")
        # print(f"Debug - 傳給 Gemini 的 prompt:\n{prompt}")
        # print("=" * 50)
        
        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=GEMINI_ADVICE_TEMPERATURE,
                    max_output_tokens=GEMINI_ADVICE_MAX_TOKENS,
                )
            )
            
            if response.text:
                # 移除所有的 Markdown 格式標記
                clean_advice = response.text.replace("**", "").replace("*", "")
                return clean_advice.strip()
            else:
                return "(系統暫時無法生成回應，請稍後再試)"
                
        except Exception as e:
            print(f"Gemini API 錯誤: {e}")
            if "quota" in str(e).lower():
                return "(API 配額已用完，請稍後再試)"
            elif "permission" in str(e).lower():
                return "(API 金鑰權限不足，請檢查設定)"
            else:
                return "(系統發生錯誤，無法取得建議)"