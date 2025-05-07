# Customer Care Assistant (Q&A)

This is an interactive assistant that allows users to upload a PDF format and ask questions about it using natural language. The system uses **LangChain**, **FAISS**, and **Streamlit** to enable Retrieval-Augmented Generation (RAG) over the uploaded documents.

---

## Features

- Upload (PDF)
- Chunk the document for efficient retrieval
- Embed chunks using Hugging Face models
- Store embeddings using FAISS vector index
- Ask questions and get accurate answers using a GROQ-hosted LLM (LLaMA 3)

Project structure:

customer-care-assistant/
│
├── src/
│   ├── ui/
│   │   ├── uploader.py              # File upload & preprocessing logic
│   │   └── chatbot.py               # Chat interface (Streamlit)
│   │
│   └── utils/
│       ├── embeddings/
│       │   ├── chunker.py           # Text splitting logic
│       │   ├── embeddings.py        # Embedding model loading
│       │   └── vector_store.py      # FAISS index creation & queries
│       │
│       └── rag/
│           ├── retriever.py         # Retrieve relevant chunks from vector DB
│           └── generator.py         # Generate answers from LLM + retrieved context
│
├── resources/                       # Uploaded files (PDF, TXT, DOCX)
├── faiss_index/                     # FAISS vector store files
│
├── main.py                          # Streamlit app entry point
├── .env                             # API keys and config
├── .gitignore
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml


### Requirements

- Python 3.10
- Docker (optional but recommended)
- GROQ API key (for LLaMA 3)

### Installation (Local)

1. Clone the repo:

```bash
git clone https://github.com/MalathiKaliappn/customer_care_agent.git


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