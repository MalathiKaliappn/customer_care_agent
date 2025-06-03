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


from typing import Optional, List
from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever
from langchain.vectorstores.base import VectorStoreRetriever
from sentence_transformers import CrossEncoder
from langchain_core.runnables import Runnable

import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings as NewHuggingFaceEmbeddings  # if you upgrade
# Use new community imports as per LangChain deprecation warnings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# Load reranker model globally (only once)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")


def rerank_documents(query: str, documents: List[Document], top_n: int = 3) -> List[Document]:
    if not documents:
        return []

    pairs = [[query, doc.page_content] for doc in documents]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in reranked[:top_n]]


class RerankingRetriever(Runnable):
    def __init__(self, base_retriever, top_k: int = 3):
        self.base_retriever = base_retriever
        self.top_k = top_k

    def invoke(self, input: str, *args, **kwargs) -> List[Document]:
        initial_docs = self.base_retriever.get_relevant_documents(input)
        return rerank_documents(input, initial_docs, self.top_k)


# def create_vector_store(index_path: str = "path/to/faiss_index") -> Optional[FAISS]:
#     """
#     Load the FAISS vector store from disk safely, enabling pickle deserialization.
#     """
#     embeddings = HuggingFaceEmbeddings()
#     try:
#         vector_store = FAISS.load_local(
#             index_path,
#             embeddings,
#             allow_dangerous_deserialization=True  # Enable loading pickle safely (only if you trust the file)
#         )
#         print(f"Loaded vector store from {index_path}")
#         return vector_store
#     except Exception as e:
#         print(f"[Error] Failed to load vector store: {e}")
#         return None
    
def create_vector_store(index_path: str) -> Optional[FAISS]:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    try:
        vector_store = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Debug: Check what type of object was loaded
        print(f"Vector store type: {type(vector_store)}")
        print(f"Has as_retriever: {hasattr(vector_store, 'as_retriever')}")
        
        if not hasattr(vector_store, "as_retriever"):
            print(f"[Error] Invalid vector_store type: {type(vector_store)}")
        return vector_store
    except Exception as e:
        print(f"[Error] Failed to load vector store: {e}")
        return None

INDEX_PATH = "faiss_index"
INDEX_FILE = os.path.join(INDEX_PATH, "index.faiss")
DOCS_FILE = os.path.join(INDEX_PATH, "docs.pkl")

def build_vector_store():
    # Implement logic to build the vector store from documents
    from langchain.document_loaders import TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    print("[Info] Building FAISS index from scratch...")

    loader = TextLoader("data/tesla_docs.txt")  # adjust as needed
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(INDEX_PATH, exist_ok=True)
    vectorstore.save_local(INDEX_PATH)

    return vectorstore

def load_vector_store():
    if os.path.exists(INDEX_FILE) and os.path.exists(DOCS_FILE):
        print("[Info] Loading FAISS index from disk...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.load_local(INDEX_PATH, embeddings)
    else:
        print("[Warning] FAISS index not found. Rebuilding...")
        return build_vector_store()

def get_retriever(
    vector_store,
    k: int = 3,
    search_type: str = "similarity",
    hybrid: bool = False,
    keyword_index: Optional[VectorStoreRetriever] = None
) -> RerankingRetriever:
    if not vector_store or not hasattr(vector_store, "as_retriever"):
        raise ValueError("Provided vector_store does not have an 'as_retriever' method.")

    semantic_retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k * 3}  # fetch more for reranking
    )

    if hybrid:
        if not keyword_index:
            raise ValueError("Hybrid retrieval enabled but no keyword_index provided.")
        if not hasattr(keyword_index, "get_relevant_documents"):
            raise TypeError("Keyword index must support 'get_relevant_documents' method.")

        hybrid_retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, keyword_index],
            weights=[0.6, 0.4],
        )
        final_retriever = hybrid_retriever
    else:
        final_retriever = semantic_retriever

    return RerankingRetriever(final_retriever, top_k=k)


def test_retriever(retriever, query: str):
    print(f"Query: {query}\n{'='*40}")
    results = retriever.get_relevant_documents(query)
    for i, doc in enumerate(results, 1):
        print(f"Document {i}:")
        print(doc.page_content)
        print("-" * 20)


if __name__ == "__main__":
    vector_store = create_vector_store("C:/Users/malat/Desktop/ML/Tesla_RAG/faiss_index")
    if vector_store:
        retriever = get_retriever(vector_store, k=5)
        test_retriever(retriever, "What is LangChain?")
