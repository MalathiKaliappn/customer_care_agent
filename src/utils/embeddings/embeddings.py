import os
from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings_model(model_name: str = "sentence-transformers/all-mpnet-base-v2") -> HuggingFaceEmbeddings:
    """
    Returns a HuggingFace embedding model instance.

    Args:
        model_name (str): The name of the sentence transformer model to use.

    Returns:
        HuggingFaceEmbeddings: The embedding model instance.
    """
    # Determine whether to use CUDA (GPU) or CPU based on environment variable
    device = "cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu"
    
    # Return the HuggingFaceEmbeddings model with the specified model and device
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})
