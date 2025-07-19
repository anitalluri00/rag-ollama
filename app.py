import os
import streamlit as st
from pathlib import Path
import torch
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader, UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer

# Ensure folders exist
Path("uploads").mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)

st.set_page_config(page_title="RAG with Llama 3", page_icon="📄", layout="wide")
st.title("📄 RAG Document Chatbot using Llama 3.2:1B")
st.markdown("Upload documents (PDF, CSV, TXT, XLSX) and ask questions. Answers are generated using Llama 3.2:1B with RAG.")

vectorstore = retriever = qa_chain = None

# Load Llama 3.2:1B Model
@st.cache_resource
def load_llama():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
    return pipe

# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

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
    embedder = load_embedding_model()
    doc_texts = [doc.page_content for doc in docs]
    embeddings = embedder.encode(doc_texts)
    vectorstore = FAISS.from_embeddings(embeddings, docs)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return vectorstore, retriever

def get_qa_chain(retriever):
    pipe = load_llama()
    template = """Use the following context to answer the question. 
If you don't know, say you don't know. Be concise.

Context:
{context}

Question:
{question}

Answer:"""
    prompt = PromptTemplate.from_template(template)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    def qa_func(inputs):
        prompt_text = prompt.format(context=inputs["context"], question=inputs["question"])
        return pipe(prompt_text, max_new_tokens=200)[0]['generated_text']
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | qa_func
        | StrOutputParser()
    )

# Upload section
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
            vectorstore, retriever = get_retriever(docs)
            qa_chain = get_qa_chain(retriever)
            st.success("File processed and embeddings created!")
        except Exception as e:
            st.error(f"Error: {e}")

if qa_chain:
    st.subheader("💬 Ask a Question")
    query = st.text_input("Type your question:")
    if st.button("Get Answer") and query:
        try:
            result = qa_chain.invoke({"query": query})
            st.success(result)
        except Exception as e:
            st.error(f"Failed to answer: {e}")
else:
    st.info("Please upload a file to begin.")
