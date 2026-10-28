from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DocumentInfo(BaseModel):
    document_id: UUID
    filename: str
    pages: int
    chunks: int


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    document_id: UUID
    question: str = Field(min_length=1, max_length=2000)


class Source(BaseModel):
    chunk_id: int
    page: int
    text: str
    similarity: float


class ChatResponse(BaseModel):
    answer: str
    status: Literal["answered", "no_context"]
    sources: list[Source]
