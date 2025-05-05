# Customer Care Assistant (Q&A)

This is an interactive assistant that allows users to upload a PDF format and ask questions about it using natural language. The system uses **LangChain**, **FAISS**, and **Streamlit** to enable Retrieval-Augmented Generation (RAG) over the uploaded documents.

---

## Features

- Upload (PDF)
- Chunk the document for efficient retrieval
- Embed chunks using Hugging Face models
- Store embeddings using FAISS vector index
- Ask questions and get accurate answers using a GROQ-hosted LLM (LLaMA 3)

---

## Project Structure

project-root/
├── src/
│ ├── main.py # Streamlit entry point
│ ├── pages/ # Streamlit pages
│ │ ├── File_Upload.py # PDF upload & embedding
│ │ └── Chatbot.py # Chat interface
│ ├── jobs/
│ │ └── kb_upload.py # CLI-based offline ingestion job (optional)
│ └── utils/
│ └── rag/
│ └── embeddings/
│ ├── chunker.py
│ ├── embeddings.py
│ ├── generation.py
│ ├── retriever.py
│ └── vector_store.py
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
└── README.md

### Requirements

- Python 3.10
- Docker (optional but recommended)
- GROQ API key (for LLaMA 3)

### Installation (Local)

1. Clone the repo:

```bash
git clone https://github.com/MalathiKaliappn/customer_care_agent/tree/developer


### Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

### Install Dependencies

pip install -r requirements.txt

### Set API Key
export GROQ_API_KEY=your_groq_key_here  # or use a .env file


### Run the App
streamlit run src/main.py

## Run with Docker
### 1. Set your API key in .env
GROQ_API_KEY=your_groq_key_here

### 2.Run via Docker Compose:
docker-compose up --build

http://localhost:8501 will launch the app