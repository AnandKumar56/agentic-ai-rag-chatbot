import re

import json

from typing import TypedDict, List

from langgraph.graph import StateGraph, START, END

from openai import OpenAI

from pinecone import Pinecone

from src.config import *

client = OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)

pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(INDEX_NAME)

GEN_SYSTEM = (
    "You answer questions about ONE document: the Konverge AI eBook 'Agentic AI for Executives'. "
    "Use ONLY the numbered context passages provided. Never use outside knowledge, even for well-known facts. "
    f"If the passages do not contain the answer, reply with exactly: {REFUSAL} "
    "If they answer only part of the question, answer that part and say what the eBook does not cover. "
    "Be concise (under about 150 words, bullets for lists). After each claim cite the page like (p. 18)."
)


GEN_SYSTEM_V2 = (
    "You answer questions about ONE document: the Konverge AI eBook 'Agentic AI for Executives'. "
    "Use ONLY the context passages provided. Never use outside knowledge, even for well-known facts. "
    f"If the passages contain nothing relevant to the question, reply with exactly: {REFUSAL} "
    "Attribute each property to the concept the passages attribute it to. For example, do not describe "
    "generative AI with traits the text gives to traditional or rule-based AI. "
    "For list questions, cover every relevant item that appears in the passages, grouped sensibly. "
    "Be concise. Use bullets for lists. "
    "Cite pages as (p. N) using ONLY the page numbers in the SOURCE PAGE headers. Never cite any other number. "
    "Do not add closing remarks about what the eBook does or does not cover."
)


GEN_SYSTEM_V3 = GEN_SYSTEM_V2 + (
    " Put a citation at the end of each bullet or sentence, not only at the end of the answer. "
    "If the context lists names without describing them, list the names without adding descriptions."
)


GEN_SYSTEM_V4 = GEN_SYSTEM_V2 + (
    " Put a citation at the end of each bullet or sentence, not only at the end of the answer. "
    "If the context lists names without describing them, list the names without adding descriptions. "
    "If the context contains several distinct lists that answer the question (for example pillars, building blocks, "
    "layers), present each list under its own short heading instead of choosing only one. "
    "When comparing two concepts, state a property for a concept only if the passages attribute that property to that "
    "concept. If a contrast has no passage support for one side, leave it out."
)


GEN_SYSTEM_V5 = GEN_SYSTEM_V4 + (
    " If the question compares two concepts and the passages compare related but differently named concepts, "
    "say which comparison the document actually makes and answer with that. "
    "Use section headings only if the exact heading text appears in the passages and belongs to the content beneath it; "
    "otherwise use plain labels without inventing titles."
)


GRADE_SYSTEM_V3 = (
    "You are a strict fact-checker. You receive CONTEXT passages (each under a 'SOURCE PAGE n' header) and an ANSWER "
    "written from them. Split the answer into individual factual claims. Ignore headings, formatting, lead-in sentences "
    "that only introduce a list (for example 'The components are:'), and closing sentences that add no facts. "
    "For EACH claim return: claim (text); cited_pages (page numbers the answer cites for it, a list); "
    "evidence: a quote copied word for word from the context (one sentence or phrase, at most 25 words, no ellipses) "
    "that directly supports the claim, or null if no passage supports it; "
    "evidence_same_concept: true only if the concept that the quoted text describes is the same concept the claim is "
    "about, false if the quote describes a different concept (for example the quote describes 'traditional AI' or "
    "'rule-based systems' but the claim is about 'generative AI chatbots'). "
    "A claim that contrasts two things needs support for BOTH sides in the context. If one side is only inferred, set "
    "evidence to null. Do not give credit for claims that are merely plausible. "
    'Output JSON only: {"claims":[{"claim":"...","cited_pages":[12],"evidence":"...","evidence_same_concept":true}]}'
)


GRADE_SYSTEM_V4 = (
    "You are a strict fact-checker. You receive CONTEXT passages (each under a 'SOURCE PAGE n' header) and an ANSWER "
    "written from them. Split the answer into ATOMIC factual claims, one fact each. A sentence that contrasts two "
    "concepts (X does A, while Y does B) must be split into separate claims, one for X and one for Y, each needing its own "
    "evidence. For EACH claim return: claim; cited_pages (list of pages the answer cites for it); "
    "evidence: a quote copied word for word from the context (at most 25 words, no ellipses) that directly supports THAT "
    "claim about THAT concept, or null if no passage supports it; "
    "evidence_same_concept: true only if the quoted text is about the same concept the claim is about (for example a quote about "
    "'traditional AI' or 'rule-based systems' does not support a claim about 'generative AI chatbots'); "
    "is_content_claim: false for headings, lead-in sentences that only introduce a list, and summary sentences that add no new "
    "fact, true otherwise. Do not credit claims that are merely plausible or inferred. "
    'Output JSON only: {"claims":[{"claim":"...","cited_pages":[12],"evidence":"...","evidence_same_concept":true,"is_content_claim":true}]}'
)


W_GROUND = 0.6          # Weight of groundedness score

W_RETR = 0.4            # Weight of retrieval strength

RETR_TOP = 0.80    # raw top-1 score that counts as "fully strong" retrieval

REFUSAL_CONF = 0.0


DEBUG = True


CLAIM_DEBUG_FOR = "How does Agentic AI differ from traditional generative AI chatbots according to the text?"


class RAGState(TypedDict, total=False):
    query: str
    raw_top1: float
    chunks: List[dict]        # [{chunk_id, page, score, text}]
    sub_queries: List[str]
    answer: str



class RAGState3(TypedDict, total=False):
    query: str
    raw_top1: float
    chunks: List[dict]
    sub_queries: List[str]
    answer: str
    bad_citations: List[int]
    attempts: int
    groundedness: float
    unsupported: List[str]
    best_answer: str
    best_g: float
    final_answer: str
    confidence: float



class RAGState4(TypedDict, total=False):
    query: str
    raw_top1: float
    chunks: List[dict]
    sub_queries: List[str]
    answer: str
    bad_citations: List[int]
    attempts: int
    groundedness: float
    unsupported: List[str]
    best_answer: str
    best_g: float
    final_answer: str
    confidence: float



class RAGState5(RAGState4, total=False):
    page_votes: dict
    top_pages: List[int]



def _hits(res):
    return [{"chunk_id": m.id, "page": int(m.metadata["page"]),
             "score": float(m.score), "text": m.metadata["text"]} for m in res.matches]



def retrieve_raw(state: RAGState) -> dict:
    qv = embed_texts([state["query"]])[0]
    res = index.query(vector=qv, top_k=K_RAW, include_metadata=True, namespace=NAMESPACE)
    hits = _hits(res)
    return {"chunks": hits, "raw_top1": hits[0]["score"] if hits else 0.0}



def route_after_raw(state: RAGState) -> str:
    return "refuse" if state["raw_top1"] < GATE else "expand"



def expand_queries(question: str, n: int = 3):
    resp = client.chat.completions.create(
        model=CHAT_MODEL, temperature=0,
        messages=[
            {"role": "system", "content": (
                f"Write {n} different search queries for finding the answer to the user's question in a book about agentic AI. "
                "Each query is 4 to 10 words. Use different vocabulary in each: "
                "(1) wording that a section title in a business book would use, "
                "(2) concrete related concepts and examples, "
                "(3) a plain rephrasing of the question. "
                "Avoid the words 'agentic', 'AI' and 'eBook' unless the question is specifically about what they mean. "
                "Output one query per line, no numbering, no extra text.")},
            {"role": "user", "content": question},
        ],
    )
    lines = [re.sub(r"^[\-\d\.\)\s]+", "", l).strip() for l in resp.choices[0].message.content.splitlines()]
    return [l for l in lines if l][:n]



def expand5(state: RAGState5) -> dict:
    subs = expand_queries(state["query"], N_SUB)
    vecs = embed_texts(subs)
    chunks = list(state["chunks"])
    seen = {c["chunk_id"] for c in chunks}
    votes, best = {}, {}

    def note(h, qi):
        votes.setdefault(h["page"], set()).add(qi)
        best[h["page"]] = max(best.get(h["page"], 0.0), h["score"])

    for h in chunks:
        note(h, 0)
    for qi, v in enumerate(vecs, start=1):
        res = index.query(vector=v, top_k=K_SUB, include_metadata=True, namespace=NAMESPACE)
        for h in _hits(res):
            note(h, qi)
            if h["chunk_id"] not in seen:
                seen.add(h["chunk_id"])
                chunks.append(h)
    page_votes = {p: [len(votes[p]), round(best[p], 3)] for p in votes}
    return {"chunks": chunks, "sub_queries": subs, "page_votes": page_votes}

def page_complete(state: RAGState5) -> dict:
    ranked = sorted(state["page_votes"].items(), key=lambda kv: (-kv[1][0], -kv[1][1]))
    pages = [p for p, _ in ranked[:N_PAGES]]
    have = {c["chunk_id"] for c in state["chunks"]}
    want = [f"p{p}-c{i}" for p in pages for i in range(1, MAX_PAGE_CHUNKS + 1) if f"p{p}-c{i}" not in have]
    added = []
    if want:
        res = index.fetch(ids=want, namespace=NAMESPACE)   # ids that do not exist are simply absent
        for vid, vec in res.vectors.items():
            added.append({"chunk_id": vid, "page": int(vec.metadata["page"]),
                          "score": 0.0, "text": vec.metadata["text"]})
    return {"chunks": state["chunks"] + added, "top_pages": pages}



def cited_pages(answer: str) -> set:
    nums = set()
    for grp in re.findall(r"\(pp?\.\s*([0-9,\s\-–and]+)\)", answer):
        nums.update(int(n) for n in re.findall(r"\d+", grp))
    return nums



def _context(chunks):
    ordered = sorted(chunks, key=lambda c: (c["page"], c["chunk_id"]))   # reading order, not score order
    return "\n\n".join(f"=== SOURCE PAGE {c['page']} ===\n{c['text']}" for c in ordered)



def generate6(state: RAGState5) -> dict:
    allowed = sorted({c["page"] for c in state["chunks"]})
    user = f"Context:\n\n{_context(state['chunks'])}\n\nQuestion: {state['query']}"
    if state.get("unsupported"):
        bullets = "\n".join(f"- {u}" for u in state["unsupported"][:8])
        user += ("\n\nA fact-checker found these problems in your previous draft:\n" + bullets +
                 "\nWrite a new answer that removes unsupported statements, fixes wrong page citations, and uses only "
                 "details stated in the context. Keep every statement that was NOT flagged.")
    messages = [{"role": "system", "content": GEN_SYSTEM_V5}, {"role": "user", "content": user}]
    answer = _chat(messages).choices[0].message.content.strip()
    bad = sorted(cited_pages(answer) - set(allowed))
    if bad:
        messages += [{"role": "assistant", "content": answer},
                     {"role": "user", "content": f"Your answer cites pages {bad}, which are not in the context. "
                      f"Allowed pages: {allowed}. Rewrite citing only allowed pages."}]
        answer = _chat(messages).choices[0].message.content.strip()
    return {"answer": answer, "bad_citations": sorted(cited_pages(answer) - set(allowed)),
            "attempts": state.get("attempts", 0) + 1}



def page_texts(chunks):
    by = {}
    for c in chunks:
        by.setdefault(c["page"], []).append(c)
    out = {}
    for p, cs in by.items():
        cs = sorted(cs, key=lambda c: int(c["chunk_id"].split("-c")[1]))
        t = cs[0]["text"]
        for c in cs[1:]:
            t = _append_with_overlap(t, c["text"])          # from Cell 4b: joins overlapping chunks
        out[p] = t
    return out



def _norm(s: str) -> str:
    s = s.lower().replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", s).strip()



def _loose(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _norm(s))



def _append_with_overlap(prev: str, new: str, max_overlap: int = 200) -> str:
    """Append `new` to `prev`, skipping any start of `new` that repeats the end of `prev`."""
    for k in range(min(len(prev), len(new), max_overlap), 19, -1):
        if prev.endswith(new[:k]):
            return prev + new[k:]
    return prev + "\n" + new



def check_claims6(claims, chunks):
    pages = {p: _loose(t) for p, t in page_texts(chunks).items()}
    ok, total, issues = 0, 0, []
    for cl in claims:
        text = (cl.get("claim") or "").strip()
        if not text or text.endswith(":") or cl.get("is_content_claim") is False:
            continue
        total += 1
        ev = cl.get("evidence")
        if not ev:
            issues.append(f"{text} [no supporting evidence in the context]")
            continue
        q = _loose(ev)
        real = {p for p, t in pages.items() if q and q in t}
        if not real:
            issues.append(f"{text} [quoted evidence does not appear in the context]")
            continue
        if cl.get("evidence_same_concept") is False:
            issues.append(f"{text} [the evidence describes a different concept than the claim]")
            continue
        cp = cl.get("cited_pages") or []
        cp = [cp] if isinstance(cp, int) else cp
        if cp and not (real & set(cp)):
            issues.append(f"{text} [cites p. {sorted(cp)} but the supporting text is on p. {sorted(real)}]")
            continue
        ok += 1
    return ok, total, issues



def grade6(state: RAGState5) -> dict:
    ans = state["answer"]
    if ans.strip() == REFUSAL:
        return {"groundedness": 1.0, "unsupported": [], "best_answer": ans, "best_g": 1.0}
    user = f"CONTEXT:\n{_context(state['chunks'])}\n\nANSWER TO CHECK:\n{ans}"
    messages = [{"role": "system", "content": GRADE_SYSTEM_V4}, {"role": "user", "content": user}]
    claims, issues = [], []
    try:
        try:
            resp = _chat(messages, model=JUDGE_MODEL, response_format={"type": "json_object"})
        except Exception:
            resp = _chat(messages, model=JUDGE_MODEL)
        claims = json.loads(re.search(r"\{.*\}", resp.choices[0].message.content, re.S).group(0)).get("claims", [])
        ok, total, issues = check_claims6(claims, state["chunks"])
        g = ok / total if total else 0.5
    except Exception as e:
        print("   grader failed, using neutral score:", repr(e)[:150])
        g = 0.5
    if state.get("bad_citations"):
        g = min(g, 0.5)
    if DEBUG:
        print(f"   [grade] attempt {state.get('attempts', 0)}: {len(claims)} claims, groundedness={g:.2f}")
        for i in issues[:8]:
            print("      issue:", i)
        if state["query"] == CLAIM_DEBUG_FOR:
            for cl in claims:
                print(f"      claim: {str(cl.get('claim'))[:70]!r} | pages={cl.get('cited_pages')} | "
                      f"same={cl.get('evidence_same_concept')} | ev={str(cl.get('evidence'))[:50]!r}")
    upd = {"groundedness": g, "unsupported": issues}
    if g > state.get("best_g", -1.0):
        upd.update(best_answer=ans, best_g=g)
    return upd



def finalize(state: RAGState3) -> dict:
    ans = state.get("best_answer", state["answer"])
    if ans.strip() == REFUSAL:
        conf = REFUSAL_CONF
    else:
        strength = min(max((state["raw_top1"] - GATE) / (RETR_TOP - GATE), 0.0), 1.0)
        conf = W_GROUND * state["best_g"] + W_RETR * strength
    return {"final_answer": ans, "confidence": round(conf, 2)}



def refuse3(state: RAGState3) -> dict:
    return {"answer": REFUSAL, "final_answer": REFUSAL, "confidence": REFUSAL_CONF}



def route_after_grade4(state: RAGState4) -> str:
    return "generate" if (state["groundedness"] < GROUND_MIN and state.get("attempts", 0) < MAX_ATTEMPTS) else "finalize"



def _chat(messages, **kw):
    model = kw.pop("model", CHAT_MODEL)
    return client.chat.completions.create(model=model, temperature=0, messages=messages, **kw)



def embed_texts(texts, batch_size=64):
    vectors = []
    for i in range(0, len(texts), batch_size):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i + batch_size])
        vectors.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
    return vectors




g6 = StateGraph(RAGState5)
g6.add_node("retrieve_raw", retrieve_raw)
g6.add_node("expand", expand5)
g6.add_node("page_complete", page_complete)
g6.add_node("generate", generate6)
g6.add_node("grade", grade6)
g6.add_node("finalize", finalize)
g6.add_node("refuse", refuse3)
g6.add_edge(START, "retrieve_raw")
g6.add_conditional_edges("retrieve_raw", route_after_raw, {"refuse": "refuse", "expand": "expand"})
g6.add_edge("expand", "page_complete")
g6.add_edge("page_complete", "generate")
g6.add_edge("generate", "grade")
g6.add_conditional_edges("grade", route_after_grade4, {"generate": "generate", "finalize": "finalize"})
g6.add_edge("finalize", END)
g6.add_edge("refuse", END)
graph = g6.compile()

def query_rag(query_text: str) -> dict:
    out = graph.invoke({"query": query_text})
    unique_chunks = []
    seen_texts = set()
    for c in sorted(out.get("chunks", []), key=lambda x: -x.get("score", 0.0)):
        txt = c["text"].strip()
        if txt not in seen_texts:
            seen_texts.add(txt)
            unique_chunks.append(txt)
    return {
        "query": query_text,
        "final_answer": out.get("final_answer", out.get("answer", "")),
        "retrieved_context_chunks": unique_chunks,
        "confidence_score": out.get("confidence", 0.0)
    }

