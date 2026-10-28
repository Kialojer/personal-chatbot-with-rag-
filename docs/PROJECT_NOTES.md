# Portfolio project notes

Owner: kiarash

## Scope decision — 2026-10-28

This new repository isolates the FastAPI, RAG and LangGraph learning path from the larger AIhu application. It does not copy production configuration, provider keys or deployment infrastructure.

- Input: an API-key-authenticated PDF upload, or a question with a document UUID.
- Processing: extract text, create overlapping chunks, embed and store in memory; retrieve matching chunks and conditionally generate an answer.
- Output: document metadata, an answer with retrieved passages, or a no-context fallback.
- Failure modes: invalid PDFs, missing keys, provider errors, unknown/deleted documents and exhausted document capacity return explicit HTTP errors.
- Observability: interactive API contracts, liveness endpoint and automated error-path tests. External tracing is not configured here.

The original supplied image is a broader reference; the editable diagram describes the implemented local example. Persistence, tenant authorization and independent grounding evaluation are deliberately omitted. The similarity threshold is a starting setting rather than a measured accuracy claim.

Run and test commands are maintained in the root README. Test results and live-provider verification must be reported separately: passing mocked tests is not proof of live AI quality.

## Local verification — 2026-10-28

On Python 3.13.1: 33 offline tests passed; Ruff lint and formatting checks passed; pip reported no broken requirements. A real local Uvicorn process served the health endpoint and the expected OpenAPI security contract. The reference PNG was checked against the supplied file byte-for-byte, and the editable diagram parsed as XML with three pages.

No live OpenAI requests or production deployments were made. Actual answer quality remains unmeasured. The next review step is to study the short graph and retrieval modules before presenting the implementation in an interview.
