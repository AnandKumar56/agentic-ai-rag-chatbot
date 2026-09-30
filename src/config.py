import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.aicredits.in/v1")
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "agentic-ai-ebook")
NAMESPACE = os.environ.get("PINECONE_NAMESPACE", "ebook")

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
JUDGE_MODEL = CHAT_MODEL
PDF_URL = "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf"
PDF_PATH = "data/Ebook-Agentic-AI.pdf"
EMBED_DIM = 1536

# RAG thresholds
GATE = 0.45
GROUND_MIN = 0.80
MAX_ATTEMPTS = 3
K_RAW = 6
K_SUB = 4
N_SUB = 4
N_PAGES = 8
MAX_PAGE_CHUNKS = 8
REFUSAL = "I could not find this information in the Agentic AI eBook."
