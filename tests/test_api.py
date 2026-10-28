from uuid import uuid4

import pytest
from conftest import make_pdf
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.providers import OpenAIProvider, ProviderError
from app.rag import CHUNK_OVERLAP, CHUNK_SIZE, MAX_BYTES, RAGService, extract_chunks


def upload(client, text="The warranty lasts two years."):
    return client.post("/documents", files={"file": ("manual.pdf", make_pdf(text))})


def test_upload_chat_and_delete(setup):
    client, _, provider = setup
    response = upload(client)
    assert response.status_code == 201
    document = response.json()
    assert document["pages"] == document["chunks"] == 1
    document_id = document["document_id"]
    assert client.get("/documents").json() == [document]
    result = client.post("/chat", json={"document_id": document_id, "question": "Warranty?"})
    assert result.status_code == 200
    body = result.json()
    assert body["status"] == "answered"
    assert "two years" in body["answer"]
    assert body["sources"][0]["page"] == 1
    assert body["sources"][0]["text"] == "The warranty lasts two years."
    assert len(provider.answer_calls) == 1
    assert client.delete(f"/documents/{document_id}").status_code == 204
    assert client.get("/documents").json() == []
    before = len(provider.embedding_calls)
    assert (
        client.post("/chat", json={"document_id": document_id, "question": "Warranty?"}).status_code
        == 404
    )
    assert len(provider.embedding_calls) == before


def test_no_evidence_takes_graph_fallback_without_generation(setup):
    client, _, provider = setup
    document_id = upload(client).json()["document_id"]
    response = client.post("/chat", json={"document_id": document_id, "question": "Weather?"})
    assert response.status_code == 200
    assert response.json()["status"] == "no_context"
    assert response.json()["sources"] == []
    assert provider.answer_calls == []


def test_retrieval_is_scoped_to_selected_document(setup):
    client, _, provider = setup
    upload(client)
    shipping = upload(client, "Shipping takes five days.").json()["document_id"]
    result = client.post("/chat", json={"document_id": shipping, "question": "Warranty?"})
    assert result.json()["status"] == "no_context"
    assert provider.answer_calls == []


@pytest.mark.parametrize(
    "path, method",
    [
        ("/documents", "get"),
        ("/documents", "post"),
        (f"/documents/{uuid4()}", "delete"),
        ("/chat", "post"),
    ],
)
def test_every_data_endpoint_requires_key(setup, path, method):
    client, _, provider = setup
    client.headers.pop("X-API-Key")
    assert client.request(method, path).status_code == 401
    assert provider.embedding_calls == []


def test_wrong_key_and_public_health(setup):
    client, _, _ = setup
    client.headers["X-API-Key"] = "wrong"
    assert client.get("/documents").status_code == 401
    assert client.get("/health").json() == {"status": "ok"}


def test_missing_app_key_fails_closed(setup):
    _, service, _ = setup
    # Explicitly blank, even if the host environment happens to contain APP_API_KEY.
    settings = Settings(_env_file=None, app_api_key="", openai_api_key="")
    with TestClient(create_app(RAGService(settings, service.provider))) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/documents").status_code == 503


@pytest.mark.parametrize(
    "filename, data, status",
    [
        ("note.txt", b"plain text", 422),
        ("fake.pdf", b"not a PDF", 422),
        ("broken.pdf", b"%PDF-1.7\nbroken", 422),
        ("blank.pdf", make_pdf(""), 422),
        ("locked.pdf", make_pdf(encrypted=True), 422),
        ("long.pdf", make_pdf(pages=21), 422),
        ("verbose.pdf", make_pdf("A" * 60_001), 422),
        ("large.pdf", b"%PDF-" + b" " * MAX_BYTES, 413),
    ],
)
def test_invalid_uploads_never_reach_provider(setup, filename, data, status):
    client, _, provider = setup
    response = client.post("/documents", files={"file": (filename, data)})
    assert response.status_code == status
    assert provider.embedding_calls == []
    assert client.get("/documents").json() == []


def test_chunk_overlap_and_page_metadata():
    text = "A" * (CHUNK_SIZE + 100)
    pages, chunks = extract_chunks(make_pdf(text, pages=2))
    assert pages == 2
    assert [page for page, _ in chunks] == [1, 1, 2, 2]
    assert len(chunks[0][1]) == CHUNK_SIZE
    assert chunks[0][1][-CHUNK_OVERLAP:] == chunks[1][1][:CHUNK_OVERLAP]


@pytest.mark.parametrize("question", ["", "   ", "A" * 2001])
def test_invalid_questions_are_rejected_before_embeddings(setup, question):
    client, _, provider = setup
    result = client.post("/chat", json={"document_id": str(uuid4()), "question": question})
    assert result.status_code == 422
    assert provider.embedding_calls == []


def test_unknown_document_is_not_sent_to_provider(setup):
    client, _, provider = setup
    assert (
        client.post(
            "/chat", json={"document_id": str(uuid4()), "question": "Warranty?"}
        ).status_code
        == 404
    )
    assert provider.embedding_calls == []


def test_capacity_limit_is_checked_before_embeddings(setup):
    client, _, provider = setup
    for _ in range(10):
        assert upload(client).status_code == 201
    before = len(provider.embedding_calls)
    assert upload(client).status_code == 409
    assert len(provider.embedding_calls) == before


@pytest.mark.parametrize("vectors", [[], [[0, 0, 0]], [[1, 0]], [[float("nan"), 0, 0]]])
def test_invalid_embedding_output_does_not_store_document(setup, monkeypatch, vectors):
    client, _, provider = setup
    monkeypatch.setattr(provider, "embed", lambda texts: vectors)
    assert upload(client).status_code == 503
    assert client.get("/documents").json() == []


def test_provider_failure_does_not_partially_store_document(setup, monkeypatch):
    client, _, provider = setup

    def fail(texts):
        raise ProviderError("Embedding provider is unavailable.")

    monkeypatch.setattr(provider, "embed", fail)
    assert upload(client).status_code == 503
    assert client.get("/documents").json() == []


def test_missing_openai_key_returns_clear_error_without_network():
    settings = Settings(_env_file=None, app_api_key="test-only-access-key", openai_api_key="")
    with TestClient(create_app(RAGService(settings, OpenAIProvider(settings)))) as client:
        client.headers["X-API-Key"] = "test-only-access-key"
        assert upload(client).json() == {"detail": "OPENAI_API_KEY is not configured."}


def test_cosine_ranking_top_k_and_source_pages(setup, monkeypatch):
    _, service, provider = setup
    settings = service.settings.model_copy(update={"top_k": 2})
    rag = RAGService(settings, provider)
    monkeypatch.setattr(
        provider,
        "embed",
        lambda texts: [[3, 4, 0], [5, 0, 0], [0, 5, 0]] if len(texts) == 3 else [[7, 0, 0]],
    )
    document = rag.ingest("manual.pdf", make_pdf(pages=3))
    hits = rag.retrieve(document.document_id, "Warranty?")
    assert [hit.page for hit in hits] == [2, 1]
    assert [hit.similarity for hit in hits] == [1.0, 0.6]


def test_answer_provider_failure_is_503(setup, monkeypatch):
    client, _, provider = setup
    document_id = upload(client).json()["document_id"]

    def fail(question, sources):
        raise ProviderError("Answer provider is unavailable.")

    monkeypatch.setattr(provider, "answer", fail)
    response = client.post("/chat", json={"document_id": document_id, "question": "Warranty?"})
    assert response.status_code == 503
    assert response.json() == {"detail": "Answer provider is unavailable."}
