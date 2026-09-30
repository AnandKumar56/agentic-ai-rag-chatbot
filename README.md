# Agentic AI RAG Chatbot

This project implements a RAG-based AI Chatbot over the "Agentic AI for Executives" eBook using LangGraph, FastAPI, and Streamlit.

## Architecture

The system uses a LangGraph-based RAG pipeline (Graph v6).
1. **Retrieve Raw**: Uses cosine similarity to retrieve initial chunks from Pinecone.
2. **Expand**: Generates alternative queries for wider recall and fetches more context.
3. **Page Complete**: Fills missing context from top-voted pages.
4. **Generate**: Generates a grounded response using OpenAI models with specific citation constraints.
5. **Grade**: Evaluates atomic claims against the context and computes groundedness scores.
6. **Finalize/Refuse**: If confidence is too low or context is absent, the system gracefully refuses.

## Project Structure

```
├── app.py                   # FastAPI backend
├── streamlit_app.py         # Streamlit frontend
├── src/
│   ├── config.py            # Configuration and environment loading
│   ├── ingestion.py         # Script to download, chunk, and index the eBook
│   └── graph.py             # LangGraph definitions and query execution logic
├── tests/
│   ├── test_api.py          # Unit tests for the API
│   └── test_rag.py          # Benchmark/unit tests for RAG
├── requirements.txt         # Project dependencies
└── README.md                # This file
```

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Environment Variables**:
   Copy `.env.example` to `.env` and fill in your keys:
   ```env
   OPENAI_API_KEY=your_key
   PINECONE_API_KEY=your_key
   ```
3. **Run Ingestion**:
   ```bash
   python src/ingestion.py
   ```
4. **Run FastAPI Server**:
   ```bash
   uvicorn app:app --port 8000 &
   ```
5. **Run Streamlit Web UI**:
   ```bash
   streamlit run streamlit_app.py --server.port 8501 --server.headless true &
   ```
