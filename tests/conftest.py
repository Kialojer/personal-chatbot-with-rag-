from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.config import Settings
from app.main import create_app
from app.rag import RAGService


def make_pdf(
    text: str = "The warranty lasts two years.", pages: int = 1, encrypted: bool = False
) -> bytes:
    writer = PdfWriter()
    for _ in range(pages):
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        escaped = text.replace("\\", "\\\\").replace("(", r"\(").replace(")", r"\)")
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    if encrypted:
        writer.encrypt("test-password")
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


class FakeProvider:
    def __init__(self):
        self.embedding_calls = []
        self.answer_calls = []

    def embed(self, texts):
        self.embedding_calls.append(texts)
        return [
            [1.0, 0.0, 0.0]
            if "warranty" in text.lower()
            else [0.0, 1.0, 0.0]
            if "shipping" in text.lower()
            else [0.0, 0.0, 1.0]
            for text in texts
        ]

    def answer(self, question, sources):
        self.answer_calls.append((question, sources))
        return f"The warranty lasts two years. [{sources[0].chunk_id}]"


@pytest.fixture
def setup():
    settings = Settings(
        _env_file=None,
        app_api_key="test-only-access-key",
        openai_api_key="",
        embedding_dimensions=3,
    )
    provider = FakeProvider()
    service = RAGService(settings, provider)
    with TestClient(create_app(service)) as client:
        client.headers["X-API-Key"] = "test-only-access-key"
        yield client, service, provider
