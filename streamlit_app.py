import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/chat"

st.set_page_config(page_title="Agentic AI RAG Chatbot", layout="wide")

st.title("Agentic AI Chatbot")
st.markdown("Ask questions about the **Agentic AI for Executives** eBook.")

st.sidebar.title("Sample Queries")
samples = [
    "What is the core definition of Agentic AI as outlined in the eBook?",
    "What are the main architectural components required to build agentic systems?",
    "What real-world industry use cases for Agentic AI are discussed in the eBook?",
    "How does Agentic AI differ from traditional generative AI chatbots according to the text?",
    "What key challenges or limitations of Agentic AI are mentioned in the document?",
    "What is the capital of France?"
]

selected_query = st.sidebar.radio("Try a query:", ["(Custom)"] + samples)

query_input = st.text_input("Your Query:", value=selected_query if selected_query != "(Custom)" else "")

if st.button("Submit"):
    if not query_input.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Generating answer..."):
            try:
                response = requests.post(API_URL, json={"query": query_input}, timeout=120)
                if response.status_code == 200:
                    data = response.json()
                    st.markdown("### Answer")
                    st.write(data["final_answer"])
                    st.markdown(f"**Confidence Score:** {data['confidence_score']:.2f}")
                    
                    with st.expander("Retrieved Context Chunks"):
                        for i, chunk in enumerate(data.get("retrieved_context_chunks", [])):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.write(chunk)
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")
