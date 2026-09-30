import os
from dotenv import load_dotenv
load_dotenv()
import json
from src.graph import query_rag

queries = [
    "What is the core definition of Agentic AI as outlined in the eBook?",
    "What are the main architectural components required to build agentic systems?",
    "What real-world industry use cases for Agentic AI are discussed in the eBook?",
    "How does Agentic AI differ from traditional generative AI chatbots according to the text?",
    "What key challenges or limitations of Agentic AI are mentioned in the document?",
    "Who won the 2022 FIFA World Cup?"
]

results = []
for q in queries:
    print(f"Querying: {q}")
    res = query_rag(q)
    results.append(res)
    print(f"Confidence: {res['confidence_score']}\n")

with open('benchmark_output.json', 'w') as f:
    json.dump(results, f, indent=2)
