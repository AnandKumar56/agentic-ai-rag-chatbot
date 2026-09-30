import re
import time
import requests
from pathlib import Path
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

from src.config import (
    PDF_URL, PDF_PATH, INDEX_NAME, NAMESPACE, EMBED_DIM,
    OPENAI_API_KEY, PINECONE_API_KEY, BASE_URL, EMBED_MODEL
)

def download_pdf():
    pdf_path = Path(PDF_PATH)
    pdf_path.parent.mkdir(exist_ok=True)
    if not pdf_path.exists():
        r = requests.get(PDF_URL, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        pdf_path.write_bytes(r.content)
    return pdf_path

def clean_text(text: str) -> str:
    text = text.replace("\ufffd", "- ")
    text = re.sub(r"(\w)\n-(\w)", r"\1-\2", text)
    kept = []
    for line in text.splitlines():
        s = line.strip()
        if s.upper() == "AGENTIC AI FOR EXECUTIVES":
            continue
        if re.fullmatch(r"\d{1,3}", s):
            continue
        kept.append(s)
    text = "\n".join(kept)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def reflow(text: str) -> str:
    out = []
    for line in text.split("\n"):
        prev = out[-1] if out else None
        starts_new = (
            prev is None
            or prev == ""
            or line == ""
            or line.startswith("- ")
            or re.match(r"^\d+(\.\d+)*\.?\s+\S", line)
        )
        if starts_new:
            out.append(line)
        else:
            out[-1] = prev + " " + line
    return "\n".join(out)

def _append_with_overlap(prev: str, new: str, max_overlap: int = 200) -> str:
    for k in range(min(len(prev), len(new), max_overlap), 19, -1):
        if prev.endswith(new[:k]):
            return prev + new[k:]
    return prev + "\n" + new

def merge_tiny(chunks, min_len=150):
    merged = []
    for c in chunks:
        same_page = bool(merged) and merged[-1].metadata["page"] == c.metadata["page"]
        if same_page and len(c.page_content) < min_len:
            merged[-1].page_content = _append_with_overlap(merged[-1].page_content, c.page_content)
        else:
            merged.append(c)
    result = []
    for i, c in enumerate(merged):
        if (len(c.page_content) < min_len and i + 1 < len(merged)
                and merged[i + 1].metadata["page"] == c.metadata["page"]):
            merged[i + 1].page_content = c.page_content + "\n" + merged[i + 1].page_content
        else:
            result.append(c)
    return result

def make_chunks(docs, size=800, overlap=100, min_len=150):
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    reflowed = [Document(page_content=reflow(d.page_content), metadata=dict(d.metadata)) for d in docs]
    chunks = merge_tiny(splitter.split_documents(reflowed), min_len=min_len)
    counter = {}
    for c in chunks:
        p = c.metadata["page"]
        counter[p] = counter.get(p, 0) + 1
        c.metadata["chunk_id"] = f"p{p}-c{counter[p]}"
    return chunks

def extract_and_chunk():
    pdf_path = download_pdf()
    reader = PdfReader(str(pdf_path))
    docs = []
    for idx, page in enumerate(reader.pages):
        txt = clean_text(page.extract_text() or "")
        if len(txt) < 80:
            continue
        docs.append(Document(
            page_content=txt,
            metadata={"source": pdf_path.name, "page": idx + 1}
        ))
    chunks = make_chunks(docs)
    return chunks

def ensure_index(pc, name=INDEX_NAME, dim=EMBED_DIM):
    if name not in pc.list_indexes().names():
        pc.create_index(
            name=name, dimension=dim, metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    while not pc.describe_index(name).status["ready"]:
        time.sleep(2)
    return pc.Index(name)

def embed_texts(texts, client, batch_size=64):
    vectors = []
    for i in range(0, len(texts), batch_size):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i + batch_size])
        vectors.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
    return vectors

def upsert_chunks(index, chunks, vectors, batch_size=100):
    records = [
        {
            "id": c.metadata["chunk_id"],
            "values": v,
            "metadata": {
                "text": c.page_content,
                "page": c.metadata["page"],
                "source": c.metadata["source"],
                "chunk_id": c.metadata["chunk_id"],
            },
        }
        for c, v in zip(chunks, vectors)
    ]
    for i in range(0, len(records), batch_size):
        index.upsert(vectors=records[i:i + batch_size], namespace=NAMESPACE)

def ingest():
    chunks = extract_and_chunk()
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = ensure_index(pc)
    
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)
    texts = [c.page_content for c in chunks]
    vectors = embed_texts(texts, client)
    upsert_chunks(index, chunks, vectors)
    print(f"Ingested {len(chunks)} chunks.")

if __name__ == "__main__":
    ingest()
