# def get_retriever(vector_store, k=3):
#     return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})


# def get_retriever(vector_store, k=3, search_type="similarity"):
#     """
#     Creates a retriever from the provided vector store to fetch similar documents.
    
#     Args:
#         vector_store: The FAISS or other vector store instance containing document embeddings.
#         k (int): The number of similar documents to retrieve (default: 3).
#         search_type (str): The search type used for similarity (default: "similarity").

#     Returns:
#         The retriever for querying the vector store.
#     """
#     if not hasattr(vector_store, "as_retriever"):
#         raise ValueError("The provided vector_store does not have an 'as_retriever' method.")

#     # Create and return the retriever
#     return vector_store.as_retriever(search_type=search_type, search_kwargs={"k": k})


from langchain.retrievers import EnsembleRetriever
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.schema import Document
from typing import Optional, Union

def get_retriever(
    vector_store,
    k: int = 3,
    search_type: str = "similarity",
    hybrid: bool = False,
    keyword_index: Optional[VectorStoreRetriever] = None
) -> Union[VectorStoreRetriever, EnsembleRetriever]:
    """
    Create a retriever from the vector store with optional hybrid retrieval.

    Args:
        vector_store: Vector store with 'as_retriever' method (e.g., FAISS index).
        k (int): Number of documents to retrieve.
        search_type (str): Type of semantic search ("similarity", "mmr", etc.).
        hybrid (bool): Whether to combine semantic + keyword retrievers.
        keyword_index: Optional keyword retriever supporting 'get_relevant_documents'.

    Returns:
        A retriever object for querying.
    """
    if not hasattr(vector_store, "as_retriever"):
        raise ValueError("Provided vector_store does not have an 'as_retriever' method.")

    semantic_retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k}
    )

    if hybrid:
        if not keyword_index:
            raise ValueError("Hybrid retrieval enabled but no keyword_index provided.")
        if not hasattr(keyword_index, "get_relevant_documents"):
            raise TypeError("Keyword index must support 'get_relevant_documents' method.")

        # Combine semantic and keyword retrievers with weighted ensemble
        hybrid_retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, keyword_index],
            weights=[0.6, 0.4],  # Adjust weights to tune importance
        )
        return hybrid_retriever

    return semantic_retriever


def test_retriever(retriever, query: str):
    """
    Helper function to query retriever and print retrieved documents.
    Useful for debugging retrieval quality.

    Args:
        retriever: Retriever instance to query.
        query (str): Query string.
    """
    print(f"Query: {query}\n{'='*40}")
    results = retriever.get_relevant_documents(query)
    for i, doc in enumerate(results, 1):
        print(f"Document {i}:")
        print(doc.page_content)
        print("-" * 20)
