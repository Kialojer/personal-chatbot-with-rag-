import json

import httpx
import pytest
from openai import OpenAI

from app.config import Settings
from app.providers import OpenAIProvider, ProviderError
from app.schemas import Source


def test_openai_adapter_over_mock_http():
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": "text-embedding-3-small",
                    "data": [
                        {"object": "embedding", "index": 1, "embedding": [0, 1, 0]},
                        {"object": "embedding", "index": 0, "embedding": [1, 0, 0]},
                    ],
                    "usage": {"prompt_tokens": 8, "total_tokens": 8},
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "object": "response",
                "created_at": 0,
                "model": "gpt-4o-mini",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_test",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Two years. [1]",
                                "annotations": [],
                            }
                        ],
                    }
                ],
            },
        )

    settings = Settings(_env_file=None, openai_api_key="")
    provider = OpenAIProvider(settings)
    with OpenAI(
        api_key="test-only-not-a-real-key",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    ) as client:
        provider.client = client
        assert provider.embed(["first", "second"]) == [[1, 0, 0], [0, 1, 0]]
        answer = provider.answer(
            "Warranty?",
            [
                Source(
                    chunk_id=1,
                    page=1,
                    text="Two years.",
                    similarity=1,
                )
            ],
        )
    assert answer == "Two years. [1]"
    assert requests[0][1]["dimensions"] == 256
    assert requests[0][1]["model"] == "text-embedding-3-small"
    assert requests[1][1]["model"] == "gpt-4o-mini"
    assert requests[1][1]["store"] is False
    assert "[1] (page 1)\nTwo years." in requests[1][1]["input"]


def test_openai_error_is_sanitized_and_not_retried():
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(
            429,
            json={
                "error": {
                    "message": "Private provider diagnostics",
                    "type": "rate_limit_error",
                }
            },
        )

    provider = OpenAIProvider(Settings(_env_file=None, openai_api_key=""))
    with OpenAI(
        api_key="test-only-not-a-real-key",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    ) as client:
        provider.client = client
        with pytest.raises(ProviderError, match="^Embedding provider is unavailable.$"):
            provider.embed(["sample"])
    assert len(calls) == 1
