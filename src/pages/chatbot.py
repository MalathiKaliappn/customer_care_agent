# import os
# import streamlit as st
# from dotenv import load_dotenv

# from customer_care_agent.src.utils.rag.embeddings.embeddings import get_embeddings_model
# from customer_care_agent.src.utils.rag.embeddings.vector_store import load_vector_store
# from langchain.prompts import PromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain.chat_models import init_chat_model
# from langchain_core.output_parsers import StrOutputParser

# # Load environment variables
# load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# # Initialize LLM model
# model = init_chat_model("llama3-8b-8192", model_provider="groq", api_key=GROQ_API_KEY)

# # Streamlit UI
# st.set_page_config(page_title="Tesla Chatbot", layout="wide")
# st.title("Tesla Manual Q&A")
# st.markdown("Ask any question based on the Tesla Model owner's manual.")

# # Load embeddings and FAISS index
# try:
#     embeddings = get_embeddings_model()
#     vector_store = load_vector_store(embeddings)
# except Exception as e:
#     st.error("FAISS index not found at 'faiss_index/index.faiss'. Please run the uploader page to generate the index first.")
#     st.stop()

# # Create retriever
# retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# # Prompt template
# prompt_template_string = """
# <|system|>
# Answer the question based on the given context. If you don't know the answer, say "I don't know".

# {context}

# {question}
# """
# prompt = PromptTemplate(
#     input_variables=["context", "question"],
#     template=prompt_template_string,
# )

# # Build RAG pipeline
# llm_chain = prompt | model | StrOutputParser()
# rag_chain = {"context": retriever, "question": RunnablePassthrough()} | llm_chain

# # User query
# query = st.text_input("Ask a question:")
# if query:
#     with st.spinner("Thinking..."):
#         answer = rag_chain.invoke(query)
#         st.markdown(f"**Answer:** {answer}")

def render():
    import os
    import streamlit as st
    from dotenv import load_dotenv

    from utils.rag.embeddings.embeddings import get_embeddings_model
    from utils.rag.embeddings.vector_store import load_vector_store
    from langchain.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain.chat_models import init_chat_model
    from langchain_core.output_parsers import StrOutputParser

    # Load environment variables
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Initialize LLM model (check if Groq provider works)
    model = init_chat_model("llama3-8b-8192", model_provider="groq", api_key=GROQ_API_KEY)

    st.markdown("### Step 2: Chat with the Assistant")
     
    # Load embeddings and FAISS index
    try:
        embeddings = get_embeddings_model()
        vector_store = load_vector_store(embeddings)
    except Exception as e:
        st.error("FAISS index not found. Please run the uploader page to generate the index first.")
        st.stop()

    # Create retriever
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Prompt template
    prompt_template_string = """
    <|system|>
    Answer the question based on the given context. If you don't know the answer, say "I don't know".

    {context}

    {question}
    """
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template_string,
    )

    # Build RAG pipeline
    llm_chain = prompt | model | StrOutputParser()
    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | llm_chain

    # User query
    query = st.text_input("Ask a question:")
    if query:
        with st.spinner("Thinking..."):
            answer = rag_chain.invoke(query)
            st.markdown(f"**Answer:** {answer}")
