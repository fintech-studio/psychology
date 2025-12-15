import asyncio
import os
from typing import List, Dict, Optional, Tuple
import logging
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


class OllamaService:
    def __init__(self):

        # Ollama runs locally and does not require an API key.
        # Check reachability in async init
        self.ollama_url = os.getenv("OLLAMA_API_URL", OLLAMA_API_URL)
        self.model_name = os.getenv("OLLAMA_MODEL_NAME", OLLAMA_MODEL_NAME)
        self.api_available = False
        self._health_checked = False
        # HTTP helper (initialized in async init)
        self._http = None

    def _option_label_for_type(self, mtype: Optional[str]) -> str:
        """Return a human-friendly option type label for UI."""
        labels = {
            "single": "選擇題",
        }
        if not mtype:
            return "其他"
        return labels.get(str(mtype).lower(), "其他")

    async def init(self):
        """Perform async health check and prepare async http client.

        This function should be awaited during application startup.
        """
        # Initialize HTTP helper and perform health checks
        from services.ollama_http import OllamaHttpClient
        self._http = OllamaHttpClient(self.ollama_url)
        logger.debug("Performing Ollama health check: %s", self.ollama_url)
        ok = await self._http.health_check_model(self.model_name)
        if not ok:
            for fallback in OLLAMA_MODEL_FALLBACKS:
                ok = await self._http.health_check_model(fallback)
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
                    r = await self._http.post_json(
                        f"{self.ollama_url}/api/generate", payload)
                    if r and r.status_code == 200:
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
        if self._http is not None:
            await self._http.close()
            self._http = None

    async def _health_check_model(self, model: str) -> bool:
        """Check a single model name for availability using a ping request.

        Returns True if API responds to a simple ping generate call.
        """
        # kept for backward compatibility but delegates to http helper
        if self._http is None:
            from services.ollama_http import OllamaHttpClient
            self._http = OllamaHttpClient(self.ollama_url)
        return await self._http.health_check_model(model)

    async def _call_ollama_generate(
        self,
        prompt: str,
        model_name: str = None,
        timeout: int = 10,
    ) -> Tuple[str, bool]:
        """Call the Ollama generate endpoint and return (text, has_response).
        Uses the HTTP helper for retries and error handling.
        """
        model = model_name if model_name else self.model_name
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": OLLAMA_ADVICE_TEMPERATURE,
            "max_tokens": OLLAMA_ADVICE_MAX_TOKENS,
        }
        # delegate to HTTP helper
        resp = None
        if self._http is None:
            from services.ollama_http import OllamaHttpClient
            self._http = OllamaHttpClient(self.ollama_url)
        resp = await self._http.post_json(
            f"{self.ollama_url}/api/generate", payload, timeout=timeout
        )
        if resp is None:
            logger.error("No response from Ollama: model=%s", model)
            return "", False
        try:
            logger.debug(
                "Ollama generate endpoint returned: %s",
                resp.status_code,
            )
            logger.debug(
                "Ollama response body (first 1024 chars): %s",
                (resp.text or "")[:1024],
            )
        except Exception:
            logger.debug("Cannot print Ollama response body")
        try:
            resp.raise_for_status()
        except Exception:
            logger.exception(
                "Ollama generate returned HTTP error: status=%s body=%s",
                resp.status_code,
                (resp.text or '')[:1024],
            )
            raise
        # extract response text field(s) using utility helpers
        from services.ollama_utils import extract_text_from_response
        out, has_response_field = extract_text_from_response(resp.text)
        return out, has_response_field

    def is_api_available(self) -> bool:
        """Public accessor for API availability after health checks."""
        return self.api_available

    async def _try_generate(
        self, prompt: str, prefer_has_response: bool = False
    ) -> Tuple[str, bool]:
        """Try generating text with the configured model and fallbacks.

        If `prefer_has_response` is True, prefer a result where the model
        returns a 'response' field. On success updates `self.model_name`
        when a fallback is used and returns (text, has_response).
        """
        # Try primary model first
        text, has_resp = await self._call_ollama_generate(prompt)
        if text:
            if not prefer_has_response or has_resp:
                return text, has_resp
        # Try fallbacks
        for fallback in OLLAMA_MODEL_FALLBACKS:
            if fallback == self.model_name:
                continue
            f_text, f_has = await self._call_ollama_generate(
                prompt, model_name=fallback
            )
            if f_text:
                # If caller prefers responses with a 'response' field, only
                # accept fallbacks that provide it. Otherwise accept any text.
                if prefer_has_response and not f_has:
                    # Skip fallbacks that produced no 'response' field.
                    continue
                self.model_name = fallback
                return f_text, f_has
        return text or "", has_resp or False

    def _as_list(self, x) -> List[str]:
        """Normalize various types to a list of strings."""
        if x is None:
            return []
        if isinstance(x, list):
            return [
                str(i).strip() for i in x if str(i).strip()
            ]
        if isinstance(x, str):
            return [line.strip() for line in x.splitlines() if line.strip()]
        return [str(x)]

    def _normalize_meta(
        self, meta: Dict, qtype: str, question: str
    ) -> Tuple[Dict, str]:
        """Normalize model-provided meta and return (meta, question_text)."""
        # Minimal normalization: rely on structured `meta` from LLM.
        if not isinstance(meta, dict):
            meta = {}

        dim_map = {
            'emotion': 'emotion',
            'stress': 'stress',
            'risk': 'risk',
            'decision_impulse': 'decision',
            'term_pref': 'time',
            'decision': 'decision'
        }
        meta.setdefault('dimension', dim_map.get(qtype, 'general'))

        # Support legacy `selection` field by mapping it to `type`.
        # Only accept 'single' as supported question type; ignore deprecated
        # 'multiple' values which were removed from the UI.
        if 'selection' in meta and meta.get('selection') == 'single':
            meta['type'] = 'single'

        # Normalize/validate `type` to a supported value.
        t = str(meta.get('type') or '').lower()
        if t == 'single':
            meta['type'] = t
        else:
            # Default to single-choice when type is missing/unknown.
            meta.setdefault('type', 'single')

        # Clean options if provided for single-choice questions.
        if meta.get('type') == 'single' and isinstance(
                meta.get('options'), list):
            opts = [
                str(o).strip() for o in meta.get('options') if str(o).strip()
            ]
            if opts:
                meta['options'] = opts
            else:
                meta.pop('options', None)

        meta['option_type'] = self._option_label_for_type(meta.get('type'))
        return meta, question

    def _build_prompt(
            self, qtype: str, current_number: int, total_questions: int
            ) -> str:
        """Build a prompt for the LLM based on a question type.
        Keeps the generation constraints in one place.
        """
        # Use prompt_templates to build full prompt
        from services.prompt_templates import build_question_prompt
        return build_question_prompt(qtype, current_number, total_questions)
        # unreachable: function returns above

    async def generate_dynamic_question(self, current_number: int,
                                        total_questions: int,
                                        previous_responses: List[Dict] = None
                                        ) -> Tuple[str, Optional[Dict]]:
        """動態生成問題內容，並確保回傳能被前端辨識為單選題(single)

        此服務現在僅回傳結構化的單選題（`type: 'single'` / `options`），
        不再依賴自由文字內的任何隱含格式。
        """
        logger.debug("generate_dynamic_question called: current=%s total=%s",
                     current_number, total_questions)
        # 題型輪替：擴充為多個 dimension（情緒 / 壓力 / 風險 / 決策衝動 / 時域 / 最近壓力）
        # 使用輪替以保證問卷包含多種類型
        qtypes = [
            "emotion",
            "stress",
            "risk",
            "decision_impulse",
            "term_pref",
            "decision",
        ]
        qtype = qtypes[(current_number - 1) % len(qtypes)]

        # 如果 Ollama API 無法使用，拋出錯誤（不使用預設題目）
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
            logger.error("AI model not available: qtype=%s", qtype)
            raise RuntimeError(
                "AI model not available: cannot generate question")

        # Prompts are built in prompt_templates.build_question_prompt
        prompt = self._build_prompt(qtype, current_number, total_questions)

        try:
            # ensure `question` is defined to avoid UnboundLocalError
            question = ""
            meta = None
            # Use _try_generate to unify fallback handling and
            # JSON detection
            resp_text, _ = await self._try_generate(
                prompt, prefer_has_response=False
            )
            # extract combined text and whether response field was present
            # Try to extract JSON if model returns structured output
            from services.ollama_utils import (
                extract_json_from_text,
                extract_text_from_response,
                sanitize_text,
            )
            meta = extract_json_from_text(resp_text)
            question, _ = extract_text_from_response(resp_text)
            logger.debug(
                "generated question raw (len=%s)", len(question)
            )
            # Ensure question is sanitized
            question = sanitize_text(question)
            logger.debug(
                "generated question (len=%s): %s",
                len(question),
                (question or '')[:250],
            )
            # 移除常見的引號或多餘符號
            # question already sanitized by _sanitize_text

            # If a structured meta was returned by the LLM, normalize it.
            if meta and isinstance(meta, dict):
                meta, question = self._normalize_meta(meta, qtype, question)

            # We expect the LLM to return structured JSON `meta`.
            # Do minimal local normalization and defaults only - avoid
            # inferring types/options from the free-text `question`.
            if meta is None:
                meta = {}

            # ensure options are strings and trimmed when present
            if meta.get('options') and isinstance(meta.get('options'), list):
                meta['options'] = [
                    str(o).strip()
                    for o in meta.get('options')
                    if str(o).strip()
                ]

            # Ensure we always provide a UI-friendly option type label
            meta.setdefault(
                'option_type',
                self._option_label_for_type(meta.get('type')),
            )

            # Ensure dimension is set for UI and analytics
            dim_map = {
                'emotion': 'emotion',
                'stress': 'stress',
                'risk': 'risk',
                'decision_impulse': 'decision',
                'term_pref': 'time',
                'decision': 'decision'
            }
            meta.setdefault('dimension', dim_map.get(qtype, 'general'))

            return question, meta

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
        question_text, meta = await self.generate_dynamic_question(
            current_number, total_questions, previous_responses
        )

        # 串流顯示問題：按區段發送，減少事件數量與 IO 開銷
        chunk_size = 6
        for i in range(0, len(question_text), chunk_size):
            part = question_text[i:i + chunk_size]
            yield {"text": part, "done": False}
            await asyncio.sleep(STREAM_DELAY * chunk_size)

        # 發送完成信號
        result = {"text": "", "done": True, "question": question_text}
        if meta:
            result["meta"] = meta
        yield result

    async def generate_content(self, all_responses: List[Dict]) -> Dict:
        """生成最終建議與結構化分析回傳。
        回傳 dict 包含：profile, investor_type, stress_index, radar, recommendations
        """
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
            raise RuntimeError(
                "AI model not available: cannot generate advice"
            )

        # build basic analysis using external AnalysisService
        from services import analysisService
        profile = analysisService.compute_profile(all_responses)
        investor_type = analysisService.classify_investor(profile)
        investor_desc = analysisService.describe_investor(profile)
        time_horizon = analysisService.compute_time_horizon(all_responses)
        stress_index = analysisService.compute_stress_index(all_responses)
        radar = analysisService.compute_radar(
            profile, stress_index, time_horizon
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

        from services.prompt_templates import build_advice_prompt
        prompt = build_advice_prompt(
            avg_negative,
            avg_neutral,
            avg_positive,
            summary_lines,
            stress_index,
            time_horizon,
        )

        # Debug: 印出傳給 Ollama 的完整內容
        logger.debug("Debug - 回答數量: %s", response_count)
        logger.debug("Debug - 傳給 Ollama 的 prompt:\n%s", prompt)
        logger.debug("%s", "=" * 50)

        try:
            # Try to generate advice; prefer models that return a
            # 'response' field for more structured outputs.
            advice, has_response = await self._try_generate(
                prompt, prefer_has_response=True
            )
            if advice:
                from services.ollama_utils import (
                    sanitize_text,
                    extract_json_from_text,
                    parse_advice_json,
                )
                clean_advice = sanitize_text(str(advice))
                clean_advice = clean_advice.replace("**", "").replace("*", "")
                # Try to parse JSON advice if present in the reply
                advice_obj = None
                try:
                    advice_obj = extract_json_from_text(advice)
                except Exception:
                    advice_obj = None

                advice_json = None
                if isinstance(advice_obj, dict):
                    advice_json = parse_advice_json(advice_obj)

                # Return structured analysis with textual advice and
                # parsed JSON when available
                result = {
                    "profile": profile,
                    "investor_type": investor_type,
                    "investor_description": investor_desc,
                    "stress_index": stress_index,
                    "time_horizon": time_horizon,
                    "radar": radar,
                    "advice": clean_advice.strip(),
                    "advice_json": advice_json,
                }
                return result
            else:
                logger.debug("Debug - Ollama returned no advice text")
                return {
                    "profile": profile,
                    "investor_type": investor_type,
                    "investor_description": investor_desc,
                    "stress_index": stress_index,
                    "time_horizon": time_horizon,
                    "radar": radar,
                    "advice": "(系統暫時無法生成回應，請稍後再試)",
                }
        except Exception as e:
            logger.exception("Ollama API 錯誤: %s", e)
            err = str(e).lower()
            if "quota" in err:
                advice_text = "(API 配額已用完，請稍後再試)"
            else:
                advice_text = "(系統發生錯誤，無法取得建議)"
            return {
                "profile": profile,
                "investor_type": investor_type,
                "investor_description": investor_desc,
                "stress_index": stress_index,
                "time_horizon": time_horizon,
                "radar": radar,
                "advice": advice_text,
            }
