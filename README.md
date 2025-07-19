# RAG Document Chatbot with Llama 3.2:1B

## Features
- Upload & search PDFs, CSVs, TXT, XLSX
- Embedding & retrieval using SentenceTransformers + FAISS
- Llama 3.2:1B for answer generation
- All-in-one Python app, deployable with Docker

## Instructions

### 1. Install dependencies
pip install -r requirements.txt

### 2. Download Llama 3.2:1B weights
Follow instructions from Hugging Face (meta-llama/Meta-Llama-3-8B-Instruct) for access and model files.

### 3. Run locally
streamlit run app.py

### 4. Build & deploy with Docker
docker build -t rag-llama .

docker run -p 8501:8501 rag-llama

### 5. Usage
Upload a supported file via the sidebar, then enter your query.

---

**Note:** Running large models may require a GPU and specific system configuration.
