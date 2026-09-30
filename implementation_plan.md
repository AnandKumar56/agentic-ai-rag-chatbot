# Implementation Plan for RAG-based AI Chatbot

## 1. Dependency List
The `requirements.txt` will contain:
```
fastapi
uvicorn
streamlit
pydantic
langchain
langchain-openai
langchain-pinecone
langchain-community
langgraph
pinecone
pypdf
openai
python-dotenv
requests
pandas
pytest
httpx
```

## 2. File-by-File Extraction

### `.env.example`
Environment variable templates:
```
OPENAI_API_KEY=
PINECONE_API_KEY=
OPENAI_BASE_URL=https://api.aicredits.in/v1
PINECONE_INDEX_NAME=agentic-ai-ebook
PINECONE_NAMESPACE=ebook
```

### `src/config.py`
Centralized configuration:
- Load environment variables.
- Export constants: `EMBED_MODEL = "text-embedding-3-small"`, `CHAT_MODEL = "gpt-4o-mini"`, `GATE = 0.45` (refusal gate score), dimensions, index name, and namespace.

### `src/ingestion.py`
Document processing logic:
- `download_pdf()`: Fetches the e-book PDF from the URL.
- `clean_text()`: Cleans PDF artifacts based on the notebook logic.
- `reflow()`: Formats text to keep paragraphs intact.
- `make_chunks()`: Uses `RecursiveCharacterTextSplitter` (size 800, overlap 100).
- `embed_and_upsert()`: Embeds with OpenAI and upserts to Pinecone.

### `src/graph.py`
Core LangGraph workflow mirroring Graph v6:
- State definitions: `RAGState` extending `TypedDict`.
- Nodes:
  - `retrieve_raw`: Embed query and fetch top-k.
  - `expand`: Multi-query expansion via LLM, then retrieve sub-queries.
  - `page_complete`: Page context collation.
  - `generate`: Synthesize answers using context and `GEN_SYSTEM_V5`.
  - `grade`: Atomic claim fact-checking with LLM and citation verification.
  - `finalize`: Select best graded answer.
  - `refuse`: Return "not in the eBook" message based on `GATE` threshold.
- Edges:
  - `retrieve_raw` -> `route_after_raw` (refuse or expand).
  - `expand` -> `page_complete` -> `generate` -> `grade` -> `route_after_grade` (loop to generate or finalize).
- Export: `query_rag(query_text: str) -> dict` returning `{"final_answer": ..., "retrieved_context_chunks": ..., "confidence_score": ...}`.

### `app.py`
FastAPI backend:
- Initialize FastAPI, add CORS middleware.
- Define `ChatRequest` and `ChatResponse` via Pydantic.
- Endpoints:
  - `GET /health`
  - `POST /chat` -> delegates to `query_rag()`

### `streamlit_app.py`
Web UI:
- Sidebar with benchmark sample queries.
- Main chat interface with Streamlit chat elements.
- Display grounded answers, chunks, and confidence scores.
- Connect to `app.py` backend.

### `tests/test_api.py` & `tests/test_rag.py`
Validation suites:
- `test_api.py`: Test FastAPI `/health` and `/chat` endpoints using `FastAPI.testclient`.
- `test_rag.py`: Automated tests for Graph v6 RAG logic with benchmark queries (in-scope vs out-of-scope).

### `README.md`
Documentation:
- Architecture diagram.
- Local setup instructions (virtual environment, dependencies, `.env`).
- Commands to run FastAPI, Streamlit, and tests.

## 3. Validation Steps
1. **Testing**: Run `pytest tests/test_api.py` and `pytest tests/test_rag.py` to ensure high test coverage and functioning routes.
2. **Backend**: Start FastAPI via `uvicorn app:app --port 8000` and use `curl` to verify response structure.
3. **Frontend**: Start Streamlit via `streamlit run streamlit_app.py --server.port 8501 --server.headless true` and perform manual visual checks using benchmark queries.
4. Verify RAG accuracy for out-of-scope adversarial questions (refusals must trigger appropriately).
