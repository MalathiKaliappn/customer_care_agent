from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.utils.rag.retriever import create_vector_store, get_retriever
from src.utils.rag.generation import build_chain

class QueryRequest(BaseModel):
    query: str

# Provide the actual FAISS index path
faiss_index_path = r"C:\Users\malat\Desktop\ML\Tesla_RAG\faiss_index"
vector_store = create_vector_store(faiss_index_path)
retriever = get_retriever(vector_store)

router = APIRouter()
qa_chain = build_chain(retriever)

@router.post("/")
async def chat(query: QueryRequest):
    try:
        answer = qa_chain.run(query.query)
        return {"answer": answer}
    except Exception as e:
        print(f"[ERROR] Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
