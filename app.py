import os
import streamlit as st
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader, CSVLoader, TextLoader, UnstructuredExcelLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import OllamaEmbeddings  # Make sure you have langchain-community>=0.2.0
import ollama

st.set_page_config(page_title="RAG with Ollama", page_icon="🤖", layout="wide")
st.title("🤖 RAG Document Chatbot with Ollama")
st.markdown("Upload a PDF, CSV, TXT, or XLSX file and ask questions about its contents using an LLM running locally (via Ollama).")

# Ensure folders exist
Path("uploads").mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)

def process_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".csv":
        loader = CSVLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    elif ext in [".xls", ".xlsx"]:
        loader = UnstructuredExcelLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

def get_retriever(docs):
    embeddings = OllamaEmbeddings(model="tinyllama")  # Use "tinyllama" or another model you have
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return ret

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def ask_ollama(context, question):
    prompt = (
        "Use the following context to answer the question concisely. "
        "If you don't know the answer, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    response = ollama.chat(
        model='tinyllama',
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']

with st.sidebar:
    st.header("📤 Upload Your File")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "csv", "txt", "xlsx", "xls"])
    if uploaded_file:
        file_path = os.path.join("uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success(f"{uploaded_file.name} uploaded!")
        try:
            docs = process_file(file_path)
            retriever = get_retriever(docs)
            st.session_state["retriever"] = retriever
            st.success("File processed and embeddings created!")
        except Exception as e:
            st.error(f"Error: {e}")

retriever = st.session_state.get("retriever", None)
if retriever:
    st.subheader("💬 Ask a Question")
    query = st.text_input("Type your question:")
    if st.button("Get Answer") and query:
        try:
            related_docs = retriever.get_relevant_documents(query)
            context = format_docs(related_docs)
            answer = ask_ollama(context, query)
            st.success(answer)
        except Exception as e:
            st.error(f"Failed to answer: {e}")
else:
    st.info("Please upload a file to begin.")
