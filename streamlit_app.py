import streamlit as st
from src.graph import query_rag

st.set_page_config(page_title="Agentic AI RAG Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 Agentic AI RAG Assistant")
st.markdown("Grounded RAG Assistant based on **'Agentic AI for Executives'** by Konverge AI.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for controls and samples
with st.sidebar:
    st.header("💡 Benchmark Sample Queries")
    st.markdown("Click any question below to test the system's accuracy and grounding.")
    
    samples = [
        "What is the core definition of Agentic AI as outlined in the eBook?",
        "What are the main architectural components required to build agentic systems?",
        "What real-world industry use cases for Agentic AI are discussed in the eBook?",
        "How does Agentic AI differ from traditional generative AI chatbots according to the text?",
        "What key challenges or limitations of Agentic AI are mentioned in the document?",
        "Who won the 2022 FIFA World Cup? (Refusal Test)"
    ]
    
    for sample in samples:
        if st.button(sample, use_container_width=True):
            st.session_state.preset_query = sample

    st.divider()
    st.markdown("### System Specs")
    st.markdown("🟢 **Backend Status**: Integrated")
    st.markdown("🗄️ **Vector DB**: Pinecone Serverless")
    st.markdown("🧠 **Models**: `text-embedding-3-small`, `gpt-4o-mini`, `gpt-4o` (Judge)")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "confidence" in message:
            st.caption(f"**Confidence Score:** {message['confidence']:.2f}")
        if "chunks" in message and message["chunks"]:
            with st.expander(f"📚 Retrieved Context ({len(message['chunks'])} chunks)"):
                for i, chunk in enumerate(message["chunks"]):
                    st.markdown(f"**Chunk #{i+1}**\n{chunk}")
                    st.divider()

# Get query (from chat input or preset)
query = st.chat_input("Ask a question about Agentic AI...")

if "preset_query" in st.session_state:
    query = st.session_state.preset_query
    del st.session_state.preset_query

if query:
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Process assistant response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving -> Expanding Queries -> Generating -> Verifying Groundedness..."):
            try:
                # Call LangGraph directly instead of via FastAPI
                data = query_rag(query)
                
                ans = data.get("final_answer", "")
                conf = data.get("confidence_score", 0.0)
                chunks = data.get("retrieved_context_chunks", [])
                
                st.markdown(ans)
                st.caption(f"**Grounding Confidence Score:** {conf:.2f}")
                
                if chunks:
                    with st.expander(f"📚 Retrieved Context ({len(chunks)} chunks)"):
                        for i, chunk in enumerate(chunks):
                            st.markdown(f"**Chunk #{i+1}**\n{chunk}")
                            st.divider()
                            
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": ans,
                    "confidence": conf,
                    "chunks": chunks
                })
            except Exception as e:
                st.error(f"Execution failed: {str(e)}")
