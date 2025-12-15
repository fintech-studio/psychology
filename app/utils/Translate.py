from typing import Optional
import logging
from transformers import pipeline, Pipeline


logger = logging.getLogger(__name__)


class Translator:
    """Lightweight wrapper for Hugging Face translation pipelines.

    To avoid repeated expensive model loads, pipelines are created lazily and
    cached on the instance. The code also falls back gracefully if a model
    cannot be created.
    """

    def __init__(self, zh_en_model: str = "Helsinki-NLP/opus-mt-zh-en",
                 en_zh_model: str = "Helsinki-NLP/opus-mt-en-zh"):
        self.zh_en_model = zh_en_model
        self.en_zh_model = en_zh_model
        self._zh_en_pipeline: Optional[Pipeline] = None
        self._en_zh_pipeline: Optional[Pipeline] = None

    def _get_zh_en_pipeline(self) -> Pipeline:
        if self._zh_en_pipeline is None:
            logger.debug("Loading translation model: %s", self.zh_en_model)
            self._zh_en_pipeline = pipeline(
                "translation", model=self.zh_en_model)
        return self._zh_en_pipeline

    def _get_en_zh_pipeline(self) -> Pipeline:
        if self._en_zh_pipeline is None:
            logger.debug("Loading translation model: %s", self.en_zh_model)
            self._en_zh_pipeline = pipeline(
                "translation", model=self.en_zh_model)
        return self._en_zh_pipeline

    def translate_zn_en(self, text: str) -> str:
        """Translate Traditional/Simplified Chinese -> English."""
        try:
            translator = self._get_zh_en_pipeline()
            result = translator(text)
            return result[0].get("translation_text", "")
        except Exception as e:
            logger.error("Translation zh->en failed: %s", e)
            return text

    def translate_en_zn(self, text: str) -> str:
        """Translate English -> Traditional/Simplified Chinese."""
        try:
            translator = self._get_en_zh_pipeline()
            result = translator(text)
            return result[0].get("translation_text", "")
        except Exception as e:
            logger.error("Translation en->zh failed: %s", e)
            return text

