# Customer Care Assistant

An AI-powered assistant that enables users to upload customer care documents (PDFs), extract relevant information, and chat with a GenAI assistant grounded in those documents using a Retrieval-Augmented Generation (RAG) pipeline. Supports both a Streamlit UI and a FastAPI backend for seamless deployment and evaluation.

---

<!-- ## Features

- Upload (PDF)
- Chunk the document for efficient retrieval
- Embed chunks using Hugging Face models
- Store embeddings using FAISS vector index
- Ask questions and get aeccurate answers using a GROQ-hosted LLM (LLaMA 3) -->

Project structure:

customer-care-assistant/
│
├── src/
│   ├── ui/                           # Streamlit UI components
│   │   ├── uploader.py               # PDF upload and text extraction
│   │   └── chatbot.py                # Streamlit chatbot interface
│   │
│   ├── api/                          # FastAPI backend
│   │   ├── chat_router.py            # Chat endpoint for RAG
│   │   ├── evaluate_router.py        # Evaluation API (classical + RAGAs)
│   │   ├── upload_router.py          # Upload and text processing
│   │   ├── health.py                 # Health check route
│   │   └── fastapi_app.py            # FastAPI app instance
│   │
│   ├── utils/
│   │   ├── embeddings/
│   │   │   ├── chunker.py            # Chunking logic
│   │   │   ├── embeddings.py         # Embedding logic (OpenAI/HuggingFace)
│   │   │   └── vector_store.py       # FAISS vector store logic
│   │   │
│   │   ├── rag/
│   │   │   ├── retriever.py          # Hybrid retriever: dense + sparse
│   │   │   └── generator.py          # Response generator using LLMs (e.g., GROQ, LLaMA 3)
│   │   │
│   │   ├── evaluation/
│   │   │   ├── evaluator.py          # F1, EM, BLEU, ROUGE etc.
│   │   │   ├── ragas_evaluator.py    # RAGAs metrics (faithfulness, etc.)
│   │   │   ├── text_normalizer.py    # Normalize predictions for evaluation
│   │   │   ├── logger.py             # Logs and results tracking
│   │   │   ├── ground_truth.json     # Gold answers for classical eval
│   │   │   └── ground_truth_ragas.json # Input format for RAGAs
│
├── resources/                        # Uploaded PDFs and extracted text
├── faiss_index/                      # Persisted FAISS vector index
│
├── main.py                           # Streamlit entry point
├── evaluate.py                       # CLI: run classical and RAGAs evaluations
├── .env                              # API keys & config
├── .gitignore
├── README.md
├── requirements.txt                  # Dependencies
├── Dockerfile
├── docker-compose.yml


Features
📄 PDF Upload & Parsing
Extract text from customer care documents using PyMuPDF.

🔍 Hybrid RAG Retrieval
Combines dense (FAISS + embeddings) and sparse (keyword) retrieval.

🤖 LLM-Based Response Generation
Uses models like LLaMA 3 or GROQ to answer user queries grounded in document content.

Evaluation Suite
Supports:

Classical metrics (F1, EM, ROUGE, BLEU)

RAGAs metrics (faithfulness, relevance, answer correctness)

FastAPI Backend
Modular API with endpoints for:

/upload – upload and index documents

/chat – retrieve and generate answers

/evaluate – evaluate predictions

/health – service status

Streamlit UI
Upload documents, chat with the bot, visualize responses.



### Requirements

- Python 3.10
- Docker (optional but recommended)
- GROQ API key (for LLaMA 3)

### Installation (Local)

1. Clone the repo:

```bash
git clone https://github.com/MalathiKaliappn/customer_care_agent.git
cd customer-care-assistant
pip install -r requirements.txt


### Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

### Install Dependencies

pip install -r requirements.txt

### Set environment variables in .env:
export GROQ_API_KEY=your_groq_key_here  # or use a .env file

LANGSMITH_API_KEY=your_groq_key_here
GROQ_API_KEY=your_groq_key_here
MISTRALAI_API_KEY=your_groq_key_here
OPENAI_API_KEY=your_groq_key_here


### Run the App
streamlit run src/main.py

### Run Evaluations (CLI)
python evaluate.py --mode classical       # Classical metrics
python evaluate.py --mode ragas           # RAGAs metrics
python evaluate.py --mode all             # Both

### Run FastAPI Server
uvicorn src.api.fastapi_app:app --reload

### Evaluation Details
Supports two modes:

Classical: F1, EM, ROUGE, BLEU

RAGAs: Faithfulness, Context Precision, Context Recall, Answer Relevancy

## Run with Docker
### 1. Set your API key in .env
GROQ_API_KEY=your_groq_key_here

### 2.Run via Docker Compose:
docker-compose up --build

http://localhost:8501 will launch the app
http://localhost:8000/docs #FastAPI