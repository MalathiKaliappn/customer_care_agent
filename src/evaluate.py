# import os
# from utils.evaluation.evaluator import Evaluator  # Adjusted path to make it consistent
# from utils.rag.generation import build_chain
# from utils.rag.retriever import get_retriever
# from utils.embeddings.vector_store import load_vector_store  # You might already have this function
# from utils.embeddings.embeddings import get_embeddings_model
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv

# # Load .env file for API keys (if any)
# load_dotenv()

# # Initialize the embedding model
# embedding_model = get_embeddings_model()

# # Print current directory for debugging
# import os
# print("Current Directory:", os.getcwd())

# # Load vector store using the embedding model
# vector_store = load_vector_store(embedding_model, r"C:\Users\malat\Desktop\ML\Tesla_RAG\faiss_index")  # Use the correct directory path

# # Initialize the retriever
# retriever = get_retriever(vector_store)

# # Initialize the LLM using Groq API and LLaMA model
# groq_key = os.getenv("GROQ_API_KEY")  # Make sure this is set in .env or replace with a direct key
# llm = ChatGroq(model_name="llama3-8b-8192", api_key=groq_key)

# # Build the QA pipeline using the retriever and the LLM
# rag_chain = build_chain(model=llm, retriever=retriever)

# # Define the function that will use the pipeline to generate answers
# def generate_answer_from_pipeline(query):
#     # Ensure query is a string
#     if isinstance(query, dict):
#         query = query.get("question", "")

#     if not isinstance(query, str):
#         raise ValueError(f"Expected query to be a string, got {type(query)}")

#     print(f"Query passed to rag_chain: {query}")

#     # Pass the raw string, not a dict
#     result = rag_chain.invoke(query)

#     print(f"Result returned by rag_chain: {result}")

#     # Handle output being a dict or string
#     if isinstance(result, dict):
#         return result.get("answer", str(result))
#     return result


# # Load the ground truth Q/A pairs for evaluation
# evaluator = Evaluator()

# # Use the absolute path to the ground truth file
# ground_truth_path = r"C:\Users\malat\Desktop\ML\Tesla_RAG\customer_care_agent\src\utils\evaluation\ground_truth.json"
# ground_truth_qa = evaluator.load_ground_truth(ground_truth_path)

# # Run the evaluation against the ground truth data
# metrics = evaluator.evaluate(ground_truth_qa, generate_answer_from_pipeline)

# # Print the evaluation metrics (F1, exact match, etc.)
# print("📊 Evaluation Results:")
# for k, v in metrics.items():
#     print(f"{k}: {v:.4f}")

# import os
# import json
# import sys
# from utils.evaluation.evaluator import Evaluator
# from utils.rag.generation import build_chain
# from utils.rag.retriever import get_retriever
# from utils.embeddings.vector_store import load_vector_store
# from utils.embeddings.embeddings import get_embeddings_model
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv

# load_dotenv()

# embedding_model = get_embeddings_model()

# print("Current Directory:", os.getcwd())

# vector_store = load_vector_store(embedding_model, r"C:\Users\malat\Desktop\ML\Tesla_RAG\faiss_index")
# retriever = get_retriever(vector_store)

# groq_key = os.getenv("GROQ_API_KEY")
# llm = ChatGroq(model_name="llama3-8b-8192", api_key=groq_key)

# rag_chain = build_chain(model=llm, retriever=retriever)

# def generate_answer_from_pipeline(query):
#     if isinstance(query, dict):
#         query = query.get("question", "")
#     if not isinstance(query, str):
#         raise ValueError(f"Expected query to be a string, got {type(query)}")
#     print(f"Query passed to rag_chain: {query}")
#     result = rag_chain.invoke(query)
#     print(f"Result returned by rag_chain: {result}")
#     if isinstance(result, dict):
#         return result.get("answer", str(result))
#     return result

# def load_ground_truth_adaptive(path):
#     with open(path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     # Detect ragas-style by simple heuristic
#     if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "question" in data[0]:
#         print("Detected ragas-style ground truth format")
#         return data

#     print("Detected standard ground truth format")
#     evaluator = Evaluator()
#     return evaluator.load_ground_truth(path)

# if __name__ == "__main__":
#     if len(sys.argv) > 1:
#         ground_truth_path = sys.argv[1]
#     else:
#         # Default path if no CLI argument provided
#         ground_truth_path = r"C:\Users\malat\Desktop\ML\Tesla_RAG\customer_care_agent\src\utils\evaluation\ground_truth.json"

#     print(f"Using ground truth file: {ground_truth_path}")

#     ground_truth_qa = load_ground_truth_adaptive(ground_truth_path)

#     evaluator = Evaluator()
#     metrics = evaluator.evaluate(ground_truth_qa, generate_answer_from_pipeline)

#     print("📊 Evaluation Results:")
#     for k, v in metrics.items():
#         print(f"{k}: {v:.4f}")


import os
import sys
import json
import asyncio
from datasets import Dataset
from utils.evaluation.evaluator import Evaluator
from utils.evaluation.ragas_evaluator import evaluate_with_ragas
from utils.rag.generation import build_chain
from utils.rag.retriever import get_retriever
from utils.embeddings.vector_store import load_vector_store
from utils.embeddings.embeddings import get_embeddings_model
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

embedding_model = get_embeddings_model()
vector_store = load_vector_store(embedding_model, r"C:\Users\malat\Desktop\ML\Tesla_RAG\faiss_index")
retriever = get_retriever(vector_store)

groq_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model_name="llama3-8b-8192", api_key=groq_key)
rag_chain = build_chain(model=llm, retriever=retriever)


def generate_answer_from_pipeline(query):
    if isinstance(query, dict):
        query = query.get("question", "")
    if not isinstance(query, str):
        raise ValueError(f"Expected query to be a string, got {type(query)}")
    print(f"Query passed to rag_chain: {query}")
    result = rag_chain.invoke(query)
    print(f"Result returned by rag_chain: {result}")
    if isinstance(result, dict):
        return result.get("answer", str(result))
    return result


def load_ground_truth(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ground_truth_path = sys.argv[1]
    else:
        ground_truth_path = r"C:\Users\malat\Desktop\ML\Tesla_RAG\utils\evaluation\ground_truth_ragas.json"

    print(f"Using ground truth file: {ground_truth_path}")
    ground_truth_qa = load_ground_truth(ground_truth_path)

    if "ragas" in ground_truth_path.lower():
        print("Running RAGAs evaluation...")
        dataset = Dataset.from_list(ground_truth_qa)

        # Run async evaluate_with_ragas using asyncio.run
        ragas_metrics = asyncio.run(evaluate_with_ragas(dataset))

        print(json.dumps(ragas_metrics, indent=2))
    else:
        print("Running classical evaluation...")
        evaluator = Evaluator()
        metrics = evaluator.evaluate(ground_truth_qa, generate_answer_from_pipeline)
        print("📊 Classical Evaluation Results:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
