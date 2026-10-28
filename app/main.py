from hmac import compare_digest
from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from app.config import Settings
from app.graph import build_graph
from app.providers import OpenAIProvider, ProviderError
from app.rag import MAX_BYTES, DocumentError, RAGService
from app.schemas import ChatRequest, ChatResponse, DocumentInfo

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def create_app(service: RAGService | None = None) -> FastAPI:
    if service is None:
        settings = Settings()
        service = RAGService(settings, OpenAIProvider(settings))
    graph = build_graph(service)
    app = FastAPI(title="Personal PDF Chatbot", version="0.1.0")

    def require_key(key: Annotated[str | None, Depends(api_key_header)]) -> None:
        expected = service.settings.app_api_key.get_secret_value()
        if not expected:
            raise HTTPException(503, "APP_API_KEY is not configured.")
        if not key or not compare_digest(key.encode(), expected.encode()):
            raise HTTPException(401, "Invalid API key.")

    @app.exception_handler(DocumentError)
    async def document_error(request: Request, exc: DocumentError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

    @app.exception_handler(ProviderError)
    async def provider_error(request: Request, exc: ProviderError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.get("/health", tags=["health"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    protected = [Depends(require_key)]

    @app.post(
        "/documents",
        response_model=DocumentInfo,
        status_code=201,
        dependencies=protected,
        tags=["documents"],
    )
    def upload(file: Annotated[UploadFile, File()]) -> DocumentInfo:
        try:
            return service.ingest(file.filename or "", file.file.read(MAX_BYTES + 1))
        finally:
            file.file.close()

    @app.get(
        "/documents", response_model=list[DocumentInfo], dependencies=protected, tags=["documents"]
    )
    def documents() -> list[DocumentInfo]:
        return service.documents()

    @app.delete(
        "/documents/{document_id}", status_code=204, dependencies=protected, tags=["documents"]
    )
    def delete(document_id: UUID) -> Response:
        service.delete(document_id)
        return Response(status_code=204)

    @app.post("/chat", response_model=ChatResponse, dependencies=protected, tags=["chat"])
    def chat(body: ChatRequest) -> ChatResponse:
        result = graph.invoke({"document_id": body.document_id, "question": body.question})
        return ChatResponse(
            answer=result["answer"], status=result["status"], sources=result["sources"]
        )

    return app


app = create_app()
