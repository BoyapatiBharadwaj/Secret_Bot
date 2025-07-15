import streamlit as st
import os
from utils import load_and_split_pdf
from rag_chain import build_vector_store, get_rag_chain

st.set_page_config(page_title="📄 RAG Chatbot", layout="wide")
st.title("🤖 PDF RAG Chatbot with Mistral")

# Initialize session state
if "rag" not in st.session_state:
    st.session_state.rag = None

# File uploader
uploaded_file = st.file_uploader("📤 Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    os.makedirs("data", exist_ok=True)  # ✅ Ensure folder exists

    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success("✅ PDF uploaded successfully!")

    # Process PDF to vector DB
    with st.spinner("🔍 Processing and embedding..."):
        docs = load_and_split_pdf(file_path)
        db = build_vector_store(docs)
        st.session_state.rag = get_rag_chain(db)
    st.success("✅ Vector DB ready! Ask your question 👇")

# Query input
query = st.text_input("💬 Ask a question from the PDF:")

if query and st.session_state.rag:
    with st.spinner("🧠 Generating answer..."):
        response = st.session_state.rag.run(query)
        st.markdown("### 📌 Answer:")
        st.write(response)
