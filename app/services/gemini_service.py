import asyncio
import os
from typing import List, Dict
import json
import logging
import httpx
from dotenv import load_dotenv
from config import (
    STREAM_DELAY,
    OLLAMA_API_URL,
    OLLAMA_MODEL_NAME,
    OLLAMA_MODEL_FALLBACKS,
    OLLAMA_ADVICE_TEMPERATURE,
    OLLAMA_ADVICE_MAX_TOKENS,
)

# 載入環境變數
load_dotenv()

logger = logging.getLogger(__name__)


class GeminiService:
    def __init__(self):
        # Ollama runs locally and does not require an API key.
        # Check reachability in async init
        self.ollama_url = os.getenv("OLLAMA_API_URL", OLLAMA_API_URL)
        self.model_name = os.getenv("OLLAMA_MODEL_NAME", OLLAMA_MODEL_NAME)
        self.api_available = False
        self._health_checked = False
        self._http_client: httpx.AsyncClient | None = None

    async def init(self):
        """Perform async health check and prepare async http client.

        This function should be awaited during application startup.
        """
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=10.0)

        try:
            # Health check: try configured model then fallbacks
            logger.debug("Performing Ollama health check: %s", self.ollama_url)
            payload = {"model": self.model_name,
                       "prompt": "ping",
                       "max_tokens": 1}
            r = await self._http_client.post(
                f"{self.ollama_url}/api/generate",
                json=payload)
            logger.debug("Health check status: %s", r.status_code)
            if r.status_code == 200:
                self.api_available = True
            else:
                self.api_available = False
        except Exception as e:
            logger.debug("Health check failed for configured model: %s", e)
            self.api_available = False

        if not self.api_available:
            logger.debug(
                "Configured model '%s' not available, trying fallbacks: %s",
                self.model_name, OLLAMA_MODEL_FALLBACKS)
            for fallback in OLLAMA_MODEL_FALLBACKS:
                try:
                    payload = {"model": fallback,
                               "prompt": "ping",
                               "max_tokens": 1}
                    r = await self._http_client.post(
                        f"{self.ollama_url}/api/generate", json=payload)
                    if r.status_code == 200:
                        self.model_name = fallback
                        self.api_available = True
                        logger.debug("Using fallback model %s", fallback)
                        break
                except Exception:
                    continue

        if self.api_available:
            logger.info(
                "Ollama local API available: %s, model: %s",
                self.ollama_url, self.model_name)
        else:
            logger.warning(
                "Ollama API not available, using local fallback responses")
        self._health_checked = True

    async def shutdown(self):
        """Close http client on shutdown"""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def _call_ollama_generate(
        self, prompt: str, model_name: str = None, timeout: int = 10
            ) -> tuple[str, bool]:
        """Call the Ollama generate endpoint for `model_name`
        and return the concatenated body text (response|thinking|text|content).
        Returns empty string if nothing can be extracted or on error.
        """
        model = model_name if model_name else self.model_name
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=timeout)
        try:
            payload = {"model": model,
                       "prompt": prompt,
                       "temperature": OLLAMA_ADVICE_TEMPERATURE,
                       "max_tokens": OLLAMA_ADVICE_MAX_TOKENS}
            # Basic retry loop for transient network issues
            retries = 3
            backoff = 0.5
            resp = None
            for attempt in range(retries):
                try:
                    resp = await self._http_client.post(
                        f"{self.ollama_url}/api/generate", json=payload)
                    break
                except (httpx.HTTPError, httpx.ConnectError) as e:
                    logger.debug("Attempt %s failed: %s", attempt + 1, e)
                    if attempt < retries - 1:
                        await asyncio.sleep(backoff * (2 ** attempt))
                        continue
                    raise
            # Basic logging
            logger.debug(
                "Ollama generate endpoint returned: %s", resp.status_code)
            try:
                logger.debug(
                    "Ollama response body (first 1024 chars): %s",
                    (resp.text or "")[:1024],
                )
            except Exception:
                logger.debug("Cannot print Ollama response body")
            if resp is None:
                raise RuntimeError("No response from Ollama API")
            try:
                resp.raise_for_status()
            except Exception:
                logger.exception(
                    "Ollama generate returned HTTP error: status=%s body=%s",
                    resp.status_code, (resp.text or '')[:1024])
                raise
            # try parse full json first
            try:
                rjson = resp.json()
            except Exception:
                rjson = None
            out = ""
            has_response_field = False
            if not rjson:
                # streaming style: JSON lines
                try:
                    lines = [line_str for line_str in resp.text.splitlines()
                             if line_str.strip()]
                    for line in lines:
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        if isinstance(obj, dict):
                            if 'response' in obj and obj.get('response'):
                                out += obj.get('response', '')
                                has_response_field = True
                            elif 'content' in obj and (
                                    isinstance(obj['content'], str)):
                                out += obj['content']
                            elif 'text' in obj:
                                out += obj.get('text', '')
                            elif 'thinking' in obj:
                                out += obj.get('thinking', '')
                except Exception:
                    pass
            else:
                if isinstance(rjson, dict):
                    if 'content' in rjson:
                        c = rjson['content']
                        if isinstance(c, list):
                            for item in c:
                                if isinstance(item, dict) and 'text' in item:
                                    out += item['text']
                                elif isinstance(item, str):
                                    out += item
                        elif isinstance(c, str):
                            out = c
                    elif 'text' in rjson:
                        out = rjson.get('text', '')
                    elif 'output' in rjson:
                        out = rjson.get('output', '')
                    elif 'response' in rjson:
                        out = rjson.get('response', '')
                        has_response_field = True
            return out.strip(), has_response_field
        except Exception:
            logger.exception("Ollama API error while calling model=%s", model)
            return "", False

    async def generate_dynamic_question(self, current_number: int,
                                        total_questions: int,
                                        previous_responses: List[Dict] = None
                                        ) -> str:
        """動態生成問題內容，並確保回傳能被前端辨識類型（MC / Likert / open）"""
        logger.debug("generate_dynamic_question called: current=%s total=%s",
                     current_number, total_questions)
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

        # 如果 Ollama API 無法使用，回傳明確格式的 fallback 題目（包含選項或 Likert 指示）
        if not self.api_available:
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
            # ensure `question` is defined to avoid UnboundLocalError
            question = ""
            # Use _call_ollama_generate to unify streaming and JSON handling
            resp_text, _ = await self._call_ollama_generate(prompt)
            resp_json = None
            try:
                resp_json = json.loads(resp_text)
            except Exception:
                resp_json = None
            # If response is JSON lines / streaming, parse by lines
            if not resp_json:
                try:
                    lines = [line_str for line_str in resp_text.splitlines()
                             if line_str.strip()]
                    for line in lines:
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        # extract probable fields from obj
                        if isinstance(obj, dict):
                            if 'response' in obj and obj.get('response'):
                                question += obj.get('response', '')
                            elif 'content' in obj and (
                                    isinstance(obj['content'], str)):
                                question += obj['content']
                            elif 'text' in obj:
                                question += obj.get('text', '')
                            elif 'thinking' in obj:
                                question += obj.get('thinking', '')
                except Exception:
                    pass
            logger.debug("Ollama JSON payload keys: %s", list(resp_json.keys())
                         if isinstance(resp_json, dict) else type(resp_json))
            # keep parsed streaming `question` if present, do not reset
            # Try to extract the text content robustly
            # from varying response shapes
            if isinstance(resp_json, dict):
                if 'content' in resp_json and isinstance(
                        resp_json['content'], list):
                    content = resp_json['content']
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and 'text' in item:
                                question += item.get('text', '')
                            elif isinstance(item, str):
                                question += item
                    elif isinstance(content, str):
                        question = content
                elif 'text' in resp_json:
                    question = resp_json.get('text', '')
                elif 'output' in resp_json:
                    question = resp_json.get('output', '')
            question = str(question).strip()
            logger.debug("generated question (len=%s): %s",
                         len(question), question[:250])
            # 移除常見的引號或多餘符號
            if question:
                question = (question.replace('"', '')
                            .replace("'", '')
                            .replace('*', ''))
                # 去掉前後多餘空白或換行
                question = ("\n".join([line.strip()
                                       for line in question.splitlines()
                                       if line.strip()]))

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
            logger.exception("動態問題生成錯誤: %s", e)
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
        if not self.api_available:
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

        # Debug: 印出傳給 Ollama 的完整內容
        print(f"Debug - 回答數量: {response_count}")
        print(f"Debug - 傳給 Ollama 的 prompt:\n{prompt}")
        print("=" * 50)

        try:
            # Call helper to get the combined advice text
            # (handles streaming & json)
            advice, has_response = await self._call_ollama_generate(prompt)
            if not advice:
                # if this model produced nothing,
                # try fallback models if available
                for fallback in OLLAMA_MODEL_FALLBACKS:
                    advice, has_response = await self._call_ollama_generate(
                        prompt, model_name=fallback)
                    print(f"Debug - tried fallback {fallback}, "
                          f"advice length: {len(advice)} "
                          f"has_response: {has_response}")
                    if advice:
                        print(f"Debug - using fallback model "
                              f"for advice: {fallback}")
                        # update service model_name to use fallback
                        # for subsequent calls
                        self.model_name = fallback
                        break
            # If the selected model only returned chain-of-thought
            # (no response field), try fallbacks
            if not has_response and advice:
                for fallback in OLLAMA_MODEL_FALLBACKS:
                    if fallback == self.model_name:
                        continue
                    f_advice, f_has_response = await (
                        self._call_ollama_generate(
                            prompt, model_name=fallback))
                    print(f"Debug - checking fallback {fallback} for "
                          f"has_response: length {len(f_advice)} "
                          f"got_response {f_has_response}")
                    if f_advice and f_has_response:
                        advice = f_advice
                        has_response = True
                        print(f"Debug - swapped to fallback model {fallback} "
                              f"because original returned no response field")
                        self.model_name = fallback
                        break
            if advice:
                clean_advice = str(advice).replace("**", "").replace("*", "")
                return clean_advice.strip()
            else:
                print("Debug - Ollama returned no advice text")
                return "(系統暫時無法生成回應，請稍後再試)"
        except Exception as e:
            logger.exception("Ollama API 錯誤: %s", e)
            err = str(e).lower()
            if "quota" in err:
                return "(API 配額已用完，請稍後再試)"
            else:
                return "(系統發生錯誤，無法取得建議)"
