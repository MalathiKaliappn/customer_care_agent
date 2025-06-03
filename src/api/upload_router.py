from fastapi import APIRouter, UploadFile, File, HTTPException
import os
import tempfile

from utils.file_processor import extract_text_by_extension
from src.utils.embeddings.chunker import chunk_text
from src.utils.embeddings.embeddings import get_embeddings_model 
from src.utils.embeddings.vector_store import create_vector_store

router = APIRouter()

@router.post("/")
async def upload_file(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]
    if suffix.lower() not in [".pdf", ".txt", ".docx"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_path = tmp_file.name
            content = await file.read()
            tmp_file.write(content)
            tmp_file.flush()

        # Extract text
        text = extract_text_by_extension(tmp_path)

        # Chunk, embed, save
        chunks = chunk_text(text)
        vectors = get_embeddings_model(chunks)
        create_vector_store(vectors, chunks)

        return {
            "message": f"{file.filename} processed and indexed successfully.",
            "total_chunks": len(chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
