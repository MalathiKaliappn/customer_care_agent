from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings_model():
    """
    Returns a HuggingFace embedding model instance.
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device": "cpu"})