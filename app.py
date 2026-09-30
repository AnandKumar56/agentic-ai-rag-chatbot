from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from src.graph import query_rag
import logging

app = FastAPI(title="Agentic AI RAG Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_context_chunks: List[str]
    confidence_score: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    logging.info(f"Received query: {request.query}")
    try:
        result = query_rag(request.query)
        return ChatResponse(**result)
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        return ChatResponse(
            final_answer=f"Error processing query: {e}",
            retrieved_context_chunks=[],
            confidence_score=0.0
        )
