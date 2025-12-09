import asyncio
from unittest.mock import patch, MagicMock

import pytest

from app.utils.Translate import Translator


@patch("app.utils.Translate.pipeline")
def test_translator_lazy_load(pipeline_mock):
    # Setup the mock pipeline to return a simple translator function
    fake_pipeline = MagicMock()
    fake_pipeline.return_value = [{"translation_text": "hello"}]
    pipeline_mock.return_value = fake_pipeline

    translator = Translator()
    # No model loaded at init
    assert translator._zh_en_pipeline is None
    assert translator._en_zh_pipeline is None

    # After call, pipelines should be loaded
    res = translator.translate_zn_en("你好")
    assert res == "hello"
    assert translator._zh_en_pipeline is not None

    # Second call should reuse the same pipeline
    res2 = translator.translate_zn_en("世界")
    assert res2 == "hello"
    assert pipeline_mock.call_count == 1


@patch("app.utils.Translate.pipeline")
def test_translator_en_zh(pipeline_mock):
    fake_pipeline = MagicMock()
    fake_pipeline.return_value = [{"translation_text": "你好"}]
    pipeline_mock.return_value = fake_pipeline

    translator = Translator()
    res = translator.translate_en_zn("hello")
    assert res == "你好"


if __name__ == "__main__":
    pytest.main()
