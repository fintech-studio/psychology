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

        if self.api_key:
            genai.configure(api_key=self.api_key)
            print("✅ Gemini API 金鑰設定成功")
            print("模型:", self.model_name)
        else:
            print("警告：未設定 GOOGLE_API_KEY，將使用模擬回應")

    async def generate_dynamic_question(self, current_number: int,
                                        total_questions: int,
                                        previous_responses: List[Dict] = None
                                        ) -> str:
        """動態生成問題內容，並確保回傳能被前端辨識類型（MC / Likert / open）"""
        # 題型輪替：1 情緒反應 (mc)，2 壓力感知 (likert)，3 風險偏好 (mc)，4 決策習慣 (mc 多選或開放)
        # 使用輪替以保證問卷包含多種類型
        qtype_cycle = (current_number - 1) % 4
        if qtype_cycle == 0:
            qtype = "emotion_mc"     # 情緒反應，選項: 冷靜觀望 / 想立刻賣出 / 加碼買進
        elif qtype_cycle == 1:
            qtype = "stress_likert"  # 壓力感知，Likert 1-5
        elif qtype_cycle == 2:
            qtype = "risk_mc"        # 風險偏好，選項: 高風險高報酬 / 穩健中報酬 / 低風險低報酬
        else:
            qtype = "decision_mc"    # 決策習慣，多選或單選描述性選項

        # 如果沒有 API Key，回傳明確格式的 fallback 題目（包含選項或 Likert 指示）
        if not self.api_key:
            if qtype == "emotion_mc":
                return "當股市短期暴跌 10% 時，您通常會怎麼做？ 冷靜觀望 / 想立刻賣出 / 加碼買進"
            if qtype == "stress_likert":
                return "在投資時，您多久會感到焦慮？請以 1 到 5 評分（1=從不，5=非常常）"
            if qtype == "risk_mc":
                return "您偏好哪種投資風格？ 高風險高報酬 / 穩健中報酬 / 低風險低報酬"
            # decision habit
            return "您通常如何做出投資決策？（可複選）列出常見做法，例如：分析公司基本面 / 聽從市場情緒 / 定期定額 / 朋友推薦"

        # 使用 Gemini 生成題目前，建立專用 prompt 強調輸出格式：
        if qtype == "emotion_mc":
            instruct = (
                "請生成一個情境式選擇題（繁體中文），要求回答者從列出的三個選項中選一個。"
                " 請以單行輸出問題，並以「 / 」分隔選項。例如：問題文字 選項A / 選項B / 選項C。"
                " 字數約 15-40 字。"
            )
        elif qtype == "stress_likert":
            instruct = (
                "請生成一個壓力感知題（繁體中文），並明確提示使用 Likert 1 到 5 評分，"
                " 請在題目中包含「1 到 5」或「1-5」等字樣以利機器判別。"
            )
        elif qtype == "risk_mc":
            instruct = (
                "請生成一個風險偏好選擇題（繁體中文），並以「 / 」分隔三個選項"
                " 題目約 10-30 字，只輸出題目與選項"
            )
        else:  # decision_mc
            instruct = (
                "請生成一個決策習慣題（繁體中文），可為單選或多選，輸出時以「 / 」或換行列出選項，"
                "字數約 15-40 字。"
            )

        prompt = f"""
你是一位理財顧問與心理評估專家。請根據下列要求生成第{current_number}題（共{total_questions}題）：
{instruct}

產出要求：
- 只輸出題目本身，不要額外說明或編號。
- 使用繁體中文。
- 若為選擇題，選項請用「 / 」分隔（例如：選項A / 選項B / 選項C）。
- 若為 Likert 題，題目中必須包含「1 到 5」或「1-5」等提示文字，方便前端判別。
- 字數控制在 10-40 字左右。
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

            question = ""
            if getattr(response, "text", None):
                question = response.text.strip()
                # 移除常見的引號或多餘符號
                question = (question.replace('"', '')
                            .replace("'", '')
                            .replace('*', ''))
                # 去掉前後多餘空白或換行
                question = ("\n".join([line.strip()
                                       for line in question.splitlines()
                                       if line.strip()]))
            else:
                question = ""

            # 若生成結果未包含預期格式，補上預設選項或 Likert 提示
            if qtype == "emotion_mc":
                if "/" not in question:
                    if question:
                        question = f"{question} 冷靜觀望 / 想立刻賣出 / 加碼買進"
                    else:
                        question = (
                            "當股市短期暴跌 10% 時，您通常會怎麼做？ "
                            "冷靜觀望 / 想立刻賣出 / 加碼買進"
                        )
            elif qtype == "stress_likert":
                if "1" not in question or "5" not in question:
                    if question:
                        question = (
                            f"{question} 請以 1 到 5 評分（1=從不，5=非常常）"
                        )
                    else:
                        question = (
                            "在投資時，您多久會感到焦慮？"
                            "請以 1 到 5 評分（1=從不，5=非常常）"
                        )
            elif qtype == "risk_mc":
                if "/" not in question:
                    question = (f"{question} 高風險高報酬 / 穩健中報酬 / 低風險低報酬"
                                if question
                                else "您偏好哪種投資風格？ 高風險高報酬 / 穩健中報酬 / 低風險低報酬")
            else:  # decision_mc
                if "/" not in question and "\n" not in question:
                    question = (f"{question} 分析公司基本面 / 聽從市場情緒 / 定期定額 / 朋友推薦"
                                if question
                                else ("您通常如何做出投資決策？ 分析公司基本面 / 聽從市場情緒 / "
                                      "定期定額 / 朋友推薦"))

            return question

        except Exception as e:
            print(f"動態問題生成錯誤: {e}")
            # 發生錯誤時使用更明確的 fallback（包含類型提示）
            if qtype == "emotion_mc":
                return "當股市短期暴跌 10% 時，您通常會怎麼做？ 冷靜觀望 / 想立刻賣出 / 加碼買進"
            if qtype == "stress_likert":
                return "在投資時，您多久會感到焦慮？請以 1 到 5 評分（1=從不，5=非常常）"
            if qtype == "risk_mc":
                return "您偏好哪種投資風格？ 高風險高報酬 / 穩健中報酬 / 低風險低報酬"
            return "您通常如何做出投資決策？ 分析公司基本面 / 聽從市場情緒 / 定期定額 / 朋友推薦"

    async def stream_question_generation(self, current_number: int,
                                         total_questions: int,
                                         previous_responses:
                                             List[Dict] = None):
        """串流方式生成問題"""
        # 首先動態生成問題內容
        question_text = await self.generate_dynamic_question(current_number,
                                                             total_questions,
                                                             previous_responses
                                                             )

        # 串流顯示問題
        for char in question_text:
            yield {"text": char, "done": False}
            await asyncio.sleep(STREAM_DELAY)  # 延遲時間

        # 發送完成信號
        yield {"text": "", "done": True, "question": question_text}

    async def generate_content(self, all_responses: List[Dict]) -> str:
        """生成最終建議（移除壓力分數聚合，僅使用情緒與問答摘要）"""
        if not self.api_key:
            return (
                "根據您的回答，建議您：1) 建立規律的壓力管理習慣 "
                "2) 尋求適當的社會支持 "
                "3) 學習正向的情緒調節技巧 "
                "4) 保持健康的生活作息"
            )

        # 構建分析摘要與情緒平均
        summary_lines = []
        total_negative = total_neutral = total_positive = 0.0
        response_count = len(all_responses)

        for i, response in enumerate(all_responses, 1):
            sentiment = response.get("sentiment", {})
            question = response.get("question", f"問題{i}")
            answer = response.get("answer", "無回答")

            total_negative += sentiment.get('negative', 0)
            total_neutral += sentiment.get('neutral', 0)
            total_positive += sentiment.get('positive', 0)

            summary_lines.append(
                f"問題{i}: {question}\n回答: {answer}\n"
                f"情緒 - 負面:{sentiment.get('negative', 0):.3f}, "
                f"正面:{sentiment.get('positive', 0):.3f}\n"
            )

        if response_count > 0:
            avg_negative = total_negative / response_count
            avg_neutral = total_neutral / response_count
            avg_positive = total_positive / response_count
        else:
            avg_negative = avg_neutral = avg_positive = 0.0

        prompt = f"""
請根據以下使用者在心理問卷中的情緒分析結果，提供個人化的心理健康建議：

整體平均情緒分析結果：
- 平均負面情緒: {avg_negative:.3f}
- 平均中性情緒: {avg_neutral:.3f}
- 平均正面情緒: {avg_positive:.3f}

詳細問答與分析：
{chr(10).join(summary_lines)}

請提供：
1. 心理狀態整體分析（基於平均分數）
2. 情緒調適技巧與壓力管理建議
3. 具體的改善方案（實務可執行）
至多 200 字，使用繁體中文回答。
        """

        # Debug: 印出傳給 Gemini 的完整內容
        print(f"Debug - 回答數量: {response_count}")
        print(f"Debug - 傳給 Gemini 的 prompt:\n{prompt}")
        print("=" * 50)

        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=GEMINI_ADVICE_TEMPERATURE,
                    max_output_tokens=GEMINI_ADVICE_MAX_TOKENS,
                )
            )

            if getattr(response, "text", None):
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
