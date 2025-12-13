import asyncio
import os
from typing import List, Dict, Optional, Tuple
import json
import re
import logging
import random
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
        # No predefined question templates: always use AI to generate content

    async def init(self):
        """Perform async health check and prepare async http client.

        This function should be awaited during application startup.
        """
        self._ensure_http_client(timeout=10)

        # Use dedicated health check logic
        # Try configured model first, then fallbacks
        logger.debug("Performing Ollama health check: %s", self.ollama_url)
        ok = await self._health_check_model(self.model_name)
        if not ok:
            for fallback in OLLAMA_MODEL_FALLBACKS:
                ok = await self._health_check_model(fallback)
                if ok:
                    self.model_name = fallback
                    break
        self.api_available = ok

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

    def _ensure_http_client(self, timeout: int = 10) -> None:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=timeout)

    async def shutdown(self):
        """Close http client on shutdown"""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def _health_check_model(self, model: str) -> bool:
        """Check a single model name for availability.

        Returns True if API responds to a simple ping generate call.
        """
        self._ensure_http_client(timeout=10)
        try:
            payload = {"model": model, "prompt": "ping", "max_tokens": 1}
            r = await self._http_client.post(
                f"{self.ollama_url}/api/generate", json=payload
            )
            logger.debug("Model %s health check status=%s", model, r.status_code)
            return r.status_code == 200
        except Exception as e:
            logger.debug("Model health check failed for %s: %s", model, e)
            return False

    async def _call_ollama_generate(
        self, prompt: str, model_name: str = None, timeout: int = 10
            ) -> tuple[str, bool]:
        """Call the Ollama generate endpoint for `model_name`
        and return the concatenated body text (response|thinking|text|content).
        Returns empty string if nothing can be extracted or on error.
        """
        model = model_name if model_name else self.model_name
        self._ensure_http_client(timeout=timeout)
        try:
            payload = {"model": model,
                       "prompt": prompt,
                       "temperature": OLLAMA_ADVICE_TEMPERATURE,
                       "max_tokens": OLLAMA_ADVICE_MAX_TOKENS}
            # Basic retry loop for transient network issues
            retries = 3
            backoff = 0.5
            max_backoff = 5.0
            resp = None
            for attempt in range(retries):
                try:
                    resp = await self._http_client.post(
                        f"{self.ollama_url}/api/generate", json=payload)
                    break
                except (httpx.RequestError, httpx.ConnectError) as e:
                    logger.debug("Attempt %s failed: %s", attempt + 1, e)
                    if attempt < retries - 1:
                        # exponential backoff with jitter
                        delay = min(max_backoff, backoff * (2 ** attempt))
                        delay = delay * (0.8 + random.random() * 0.4)
                        await asyncio.sleep(delay)
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
            # extract response text field(s) robustly using helper
            out, has_response_field = self._extract_text_from_response(resp.text)
            # return the output and whether the response field was present
            return out, has_response_field
        except Exception:
            logger.exception("Ollama API error while calling model=%s", model)
            return "", False

    def is_api_available(self) -> bool:
        """Public accessor for API availability after health checks."""
        return self.api_available

    def _extract_text_from_response(self, raw_text: str) -> Tuple[str, bool]:
        """Parse a raw response string that could be streaming JSON lines
        or plain text JSON and return a concatenated text and whether a
        'response' field was present.
        """
        if not raw_text:
            return "", False
        # Try parse as JSON object first
        try:
            obj = json.loads(raw_text)
        except Exception:
            obj = None

        out = ""
        has_response = False
        if isinstance(obj, dict):
            if 'response' in obj and obj.get('response'):
                out += obj.get('response', '')
                has_response = True
            elif 'content' in obj:
                c = obj['content']
                if isinstance(c, list):
                    for item in c:
                        if isinstance(item, dict) and 'text' in item:
                            out += item['text']
                        elif isinstance(item, str):
                            out += item
                elif isinstance(c, str):
                    out += c
            elif 'text' in obj:
                out += obj.get('text', '')
            elif 'output' in obj:
                out += obj.get('output', '')
            return out.strip(), has_response

        # If not JSON object, parse by lines and try to load JSON per line
        try:
            for line in raw_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    out += line + "\n"
                    continue
                if isinstance(item, dict):
                    if 'response' in item and item.get('response'):
                        out += item.get('response', '')
                        has_response = True
                    elif (
                        'content' in item
                        and isinstance(item['content'], str)
                    ):
                        out += item['content']
                    elif 'text' in item:
                        out += item.get('text', '')
                    elif 'thinking' in item:
                        out += item.get('thinking', '')
        except Exception:
            pass
        return out.strip(), has_response

    def _sanitize_text(self, s: Optional[str]) -> str:
        if not s:
            return ""
        txt = str(s)
        txt = txt.replace('"', '').replace("'", '').replace('*', '')
        txt = "\n".join(
            [line.strip() for line in txt.splitlines() if line.strip()]
        )
        return txt

    def _build_prompt(self, qtype: str, current_number: int, total_questions: int) -> str:
        """Build a prompt for the LLM based on a question type.
        Keeps the generation constraints in one place.
        """
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
        return f"你是一位理財顧問與心理評估專家。請根據下列要求生成第{current_number}題（共{total_questions}題）：\n{instruct}\n\n產出要求：\n- 只輸出題目本身，不要額外說明或編號。\n- 使用繁體中文。\n- 若為選擇題，選項請用「 / 」分隔（例如：選項A / 選項B / 選項C）。\n- 若為 Likert 題，題目中必須包含「1 到 5」或「1-5」等提示文字，方便前端判別。\n- 字數控制在 10-40 字左右。\n        "

    async def _repair_placeholders(self, question: str, qtype: str, attempts: int = 2) -> str:
        """Try to repair a generated question that contains placeholder options.

        This function will ask the LLM to rewrite the question, replacing
        placeholders like '選擇A' or 'A / B / C' with real options and ensuring
        the appropriate format based on qtype.
        Returns the repaired question string or raises RuntimeError on failure.
        """
        if not question:
            raise RuntimeError("Empty question to repair")
        # Build a repair prompt tailored to the qtype
        if qtype in ("emotion_mc", "risk_mc", "decision_mc"):
            repair_instruct = (
                "上面問題包含像 '選擇A' 或 'A / B / C' 的佔位符。請將其替換為三個有意義、繁體中文的選項，\n"
                "僅回傳修正後的單行題目，選項以 ' / ' 分隔，並保留原始問題意涵。"
            )
        elif qtype == "stress_likert":
            repair_instruct = (
                "上面問題未包含 Likert 1 到 5 的提示。請在題目中加入 '1 到 5' 或 '1-5' 的提示，\n"
                "並僅回傳修正後的單行題目（保持繁體中文）。"
            )
        else:
            repair_instruct = (
                "請將問題修正為合適的格式，並以繁體中文回傳，若為選擇題請用 ' / ' 分隔選項。"
            )

        base_prompt = f"請修正以下問題：\n{question}\n\n{repair_instruct}\n只回傳題目本身。"

        last_err = None
        for i in range(attempts):
            try:
                repaired_text, _ = await self._call_ollama_generate(base_prompt)
                repaired_text = self._sanitize_text(repaired_text)
                if repaired_text and not self._contains_placeholder_options(repaired_text):
                    # additional Likert check
                    if qtype == "stress_likert" and ("1" not in repaired_text or "5" not in repaired_text):
                        # not repaired properly; try again
                        last_err = RuntimeError("Repaired Likert question still missing 1-5")
                        continue
                    return repaired_text
                else:
                    last_err = RuntimeError("Repaired question still contains placeholders or is empty")
            except Exception as e:
                last_err = e
        raise RuntimeError(f"Failed to repair question: {last_err}")

    def _contains_placeholder_options(self, question: str) -> bool:
        """Detect if the provided question contains placeholder options like
        '選擇A', '選項A', 'A / B / C', 'Option A / Option B' etc.
        Returns True if placeholders are detected.
        """
        if not question or "/" not in question:
            return False
        parts = [p.strip() for p in question.split("/") if p.strip()]
        if len(parts) < 2:
            return False
        placeholder_count = 0
        for p in parts:
            # single letter options (A, B, C) or letter in parentheses or dots
            if len(p) == 1 and p.isalpha() and p.upper() in "ABCD":
                placeholder_count += 1
                continue
            # Matches '選擇A' / '選項A' / '選A'
            if any(
                p.startswith(pref) and len(p.strip()) <= 4
                for pref in ["選擇", "選項", "選"]
            ):
                placeholder_count += 1
                continue
            # Matches 'Option A' or '選項 A'
            if (
                p.lower().startswith("option")
                and any(ch.isalpha() for ch in p)
            ):
                placeholder_count += 1
                continue
        # if most or all options seem placeholders, return True
        result_bool = placeholder_count >= max(1, len(parts) // 2)
        logger.debug(
            "Placeholder detection: placeholders=%s parts=%s",
            result_bool,
            len(parts),
        )
        return result_bool

    def _strip_placeholder_options(self, question: str) -> str:
        """Remove common placeholder option tokens from a question string."""
        if not question or "/" not in question:
            return question
        # Remove tokens like '選擇A', '選項A', 'Option A'
        # and lone letters like 'A / B / C'
        # Replace them and then cleanup duplicate slashes and extra whitespace
        q = re.sub(r"(?:選擇|選項)?\s*[A-D]", "", question)
        q = re.sub(r"Option\s*[A-D]", "", q, flags=re.IGNORECASE)
        # Remove any leftover isolated letter options like 'A / B / C'
        q = re.sub(r"\b[A-D]\b", "", q)
        # Clean up repeated slashes / extra spaces
        q = re.sub(r"\s*/\s*", " / ", q).strip()
        # Remove trailing slashes or punctuation related to options
        q = re.sub(r"[\s/]*$", "", q)
        q = q.strip()
        logger.debug(
            "Stripped placeholder options: '%s' -> '%s'",
            question[:200],
            q[:200],
        )
        return q

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
        # Ensure we checked API health, try to init if not checked
        if not self._health_checked:
            try:
                await self.init()
            except Exception:
                logger.debug(
                    "Failed to init/health check; using fallback responses"
                )

        if not self.api_available:
            # Do not return predefined questions; require AI-generated content.
            logger.error("AI model not available; cannot generate question for qtype=%s", qtype)
            raise RuntimeError("AI model not available: cannot generate question")

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

        prompt = self._build_prompt(qtype, current_number, total_questions)

        try:
            # ensure `question` is defined to avoid UnboundLocalError
            question = ""
            # Use _call_ollama_generate to unify streaming and JSON handling
            resp_text, _ = await self._call_ollama_generate(prompt)
            # extract combined text and whether response field was present
            question, _ = self._extract_text_from_response(resp_text)
            logger.debug(
                "generated question raw (len=%s)", len(question)
            )
            # Ensure question is sanitized
            question = self._sanitize_text(question)
            logger.debug(
                "generated question (len=%s): %s",
                len(question),
                (question or '')[:250],
            )
            # 移除常見的引號或多餘符號
            # question already sanitized by _sanitize_text

            # Validate format but do not inject default options. Try AI-based repair on placeholders.
            if qtype == "emotion_mc":
                # If the model didn't produce comma/ slash separated options, reject
                if "/" not in question or self._contains_placeholder_options(question):
                    logger.warning("Generated question contains placeholders or missing options for emotion_mc: %s. Attempting repair.", question)
                    try:
                        question = await self._repair_placeholders(question, qtype)
                        logger.debug("Repaired emotion_mc question: %s", question)
                    except Exception as e:
                        logger.error("Repair failed for emotion_mc: %s", e)
                        # fallback: raise error (no pre-written examples)
                        raise
            elif qtype == "stress_likert":
                if "1" not in question or "5" not in question:
                    logger.warning("stress_likert missing Likert hints: %s. Attempting repair.", question)
                    try:
                        question = await self._repair_placeholders(question, qtype)
                        logger.debug("Repaired stress_likert question: %s", question)
                    except Exception as e:
                        logger.error("Repair failed for stress_likert: %s", e)
                        raise
            elif qtype == "risk_mc":
                if "/" not in question or self._contains_placeholder_options(question):
                    logger.warning("Generated question contains placeholders or missing options for risk_mc: %s. Attempting repair.", question)
                    try:
                        question = await self._repair_placeholders(question, qtype)
                        logger.debug("Repaired risk_mc question: %s", question)
                    except Exception as e:
                        logger.error("Repair failed for risk_mc: %s", e)
                        raise
            else:  # decision_mc
                if ("/" not in question and "\n" not in question) or self._contains_placeholder_options(question):
                    logger.warning("Generated question contains placeholders or missing options for decision_mc: %s. Attempting repair.", question)
                    try:
                        question = await self._repair_placeholders(question, qtype)
                        logger.debug("Repaired decision_mc question: %s", question)
                    except Exception as e:
                        logger.error("Repair failed for decision_mc: %s", e)
                        raise

            return question

        except Exception as e:
            logger.exception("動態問題生成錯誤: %s", e)
            # Do not use pre-defined fallback questions. Propagate as error.
            raise

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
        # Ensure service health checked
        if not self._health_checked:
            try:
                await self.init()
            except Exception:
                logger.debug(
                    "Failed to init/health check; using fallback responses"
                )

        if not self.api_available:
            logger.error("AI model not available; cannot generate advice")
            raise RuntimeError("AI model not available: cannot generate advice")

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
        logger.debug("Debug - 回答數量: %s", response_count)
        logger.debug("Debug - 傳給 Ollama 的 prompt:\n%s", prompt)
        logger.debug("%s", "=" * 50)

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
                    logger.debug(
                        "Fallback tried: %s len=%s has_resp=%s",
                        fallback,
                        len(advice),
                        has_response,
                    )
                    if advice:
                        logger.debug(
                            "Debug - using fallback model for advice: %s",
                            fallback,
                        )
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
                    logger.debug(
                        "Checking fallback: %s len=%s got_resp=%s",
                        fallback,
                        len(f_advice),
                        f_has_response,
                    )
                    if f_advice and f_has_response:
                        advice = f_advice
                        has_response = True
                        logger.debug(
                            "Swapped to fallback model %s because original has"
                            " no response field",
                            fallback,
                        )
                        self.model_name = fallback
                        break
            if advice:
                clean_advice = self._sanitize_text(str(advice))
                clean_advice = clean_advice.replace("**", "").replace("*", "")
                return clean_advice.strip()
            else:
                logger.debug("Debug - Ollama returned no advice text")
                return "(系統暫時無法生成回應，請稍後再試)"
        except Exception as e:
            logger.exception("Ollama API 錯誤: %s", e)
            err = str(e).lower()
            if "quota" in err:
                return "(API 配額已用完，請稍後再試)"
            else:
                return "(系統發生錯誤，無法取得建議)"
