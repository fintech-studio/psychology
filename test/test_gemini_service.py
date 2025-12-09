import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from app.services.gemini_service import GeminiService


class DummyResp:
    def __init__(self, status_code=200, text='{"text": "Hello world"}'):
        self.status_code = status_code
        self.text = text

    def json(self):
        return {"text": "Hello world"}

    def raise_for_status(self):
        if self.status_code != 200:
            raise Exception("HTTP Error")


@pytest.mark.asyncio
async def test_call_ollama_generate_single_text(monkeypatch):
    gs = GeminiService()
    # Provide a mocked AsyncClient with a post method
    mock_client = AsyncMock()
    mock_client.post.return_value = DummyResp(status_code=200, text='{"text": "Hello world"}')
    gs._http_client = mock_client

    out, has_response = await gs._call_ollama_generate("ping")
    assert out == "Hello world"
    assert has_response is False


@pytest.mark.asyncio
async def test_call_ollama_generate_lines(monkeypatch):
    gs = GeminiService()
    mock_client = AsyncMock()
    # Simulate JSON lines with response field
    lines = '\n'.join(["{\"response\": \"Hello\"}", "{\"response\": \" world\"}"])
    mock_client.post.return_value = DummyResp(status_code=200, text=lines)
    gs._http_client = mock_client

    out, has_response = await gs._call_ollama_generate("ping")
    assert out == "Hello world"
    assert has_response is True

if __name__ == "__main__":
    pytest.main()
