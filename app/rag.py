import math
from dataclasses import dataclass
from io import BytesIO
from threading import RLock
from uuid import UUID, uuid4

from pypdf import PdfReader

from app.config import Settings
from app.providers import AIProvider, ProviderError
from app.schemas import DocumentInfo, Source

MAX_BYTES = 10_000_000
MAX_PAGES = 20
MAX_CHARACTERS = 60_000
MAX_DOCUMENTS = 10
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


class DocumentError(Exception):
    def __init__(self, detail: str, status_code: int = 422) -> None:
        super().__init__(detail)
        self.status_code = status_code


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    page: int
    text: str
    vector: tuple[float, ...]


@dataclass(frozen=True)
class Document:
    info: DocumentInfo
    chunks: tuple[Chunk, ...]


def extract_chunks(data: bytes) -> tuple[int, list[tuple[int, str]]]:
    if len(data) > MAX_BYTES:
        raise DocumentError("PDF exceeds 10 MB.", 413)
    if not data.startswith(b"%PDF-"):
        raise DocumentError("Upload a valid PDF.")
    try:
        reader = PdfReader(BytesIO(data), strict=True)
        if reader.is_encrypted:
            raise DocumentError("Encrypted PDFs are not supported.")
        pages = len(reader.pages)
        if not 1 <= pages <= MAX_PAGES:
            raise DocumentError("PDF must contain 1 to 20 pages.")
        chunks: list[tuple[int, str]] = []
        characters = 0
        for number, page in enumerate(reader.pages, start=1):
            text = " ".join((page.extract_text() or "").split())
            characters += len(text)
            if characters > MAX_CHARACTERS:
                raise DocumentError("PDF exceeds 60,000 extracted characters.")
            start = 0
            while start < len(text):
                end = min(start + CHUNK_SIZE, len(text))
                chunks.append((number, text[start:end]))
                if end == len(text):
                    break
                start = end - CHUNK_OVERLAP
    except DocumentError:
        raise
    except Exception as exc:
        raise DocumentError("PDF could not be read.") from exc
    if not chunks:
        raise DocumentError("PDF has no readable text. Scanned PDFs need OCR first.")
    return pages, chunks


def unit_vector(values: list[float], dimensions: int) -> tuple[float, ...]:
    if len(values) != dimensions or not all(math.isfinite(value) for value in values):
        raise ProviderError("Embedding provider returned an invalid vector.")
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or norm == 0:
        raise ProviderError("Embedding provider returned an invalid vector.")
    return tuple(value / norm for value in values)


class RAGService:
    """One local workspace, held in memory; not a multi-tenant vector database."""

    def __init__(self, settings: Settings, provider: AIProvider) -> None:
        self.settings = settings
        self.provider = provider
        self._documents: dict[UUID, Document] = {}
        self._lock = RLock()

    def ingest(self, filename: str, data: bytes) -> DocumentInfo:
        filename = filename.replace("\\", "/").rsplit("/", 1)[-1][:255]
        if not filename.lower().endswith(".pdf"):
            raise DocumentError("Only PDF files are accepted.")
        with self._lock:
            if len(self._documents) >= MAX_DOCUMENTS:
                raise DocumentError("Delete a document before uploading another.", 409)
        pages, passages = extract_chunks(data)
        vectors = self.provider.embed([text for _, text in passages])
        if len(vectors) != len(passages):
            raise ProviderError("Embedding provider returned an unexpected vector count.")
        chunks = tuple(
            Chunk(index, page, text, unit_vector(vector, self.settings.embedding_dimensions))
            for index, ((page, text), vector) in enumerate(zip(passages, vectors), start=1)
        )
        info = DocumentInfo(document_id=uuid4(), filename=filename, pages=pages, chunks=len(chunks))
        with self._lock:
            if len(self._documents) >= MAX_DOCUMENTS:
                raise DocumentError("Document capacity reached during upload.", 409)
            self._documents[info.document_id] = Document(info, chunks)
        return info

    def documents(self) -> list[DocumentInfo]:
        with self._lock:
            return [document.info for document in self._documents.values()]

    def delete(self, document_id: UUID) -> None:
        with self._lock:
            if self._documents.pop(document_id, None) is None:
                raise DocumentError("Document not found.", 404)

    def retrieve(self, document_id: UUID, question: str) -> list[Source]:
        with self._lock:
            document = self._documents.get(document_id)
            if document is None:
                raise DocumentError("Document not found.", 404)
        vectors = self.provider.embed([question])
        if len(vectors) != 1:
            raise ProviderError("Embedding provider returned an unexpected vector count.")
        query = unit_vector(vectors[0], self.settings.embedding_dimensions)
        ranked = []
        for chunk in document.chunks:
            score = sum(a * b for a, b in zip(query, chunk.vector))
            if score >= self.settings.min_similarity:
                ranked.append(
                    Source(
                        chunk_id=chunk.chunk_id,
                        page=chunk.page,
                        text=chunk.text,
                        similarity=round(score, 6),
                    )
                )
        return sorted(ranked, key=lambda source: source.similarity, reverse=True)[
            : self.settings.top_k
        ]
