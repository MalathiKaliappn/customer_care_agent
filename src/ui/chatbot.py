import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
from dotenv import load_dotenv

# Updated imports based on the refactor
from utils.embeddings.embeddings import get_embeddings_model
from utils.embeddings.vector_store import load_vector_store
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser

def render():
    # Load environment variables
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not found in environment. Please set it in a .env file.")
        st.stop()

    # Initialize LLM model (Groq-based)
    model = init_chat_model("llama3-8b-8192", model_provider="groq", api_key=GROQ_API_KEY)

    st.markdown("### Step 2: Chat with the Assistant")

    # Load embeddings and FAISS index
    try:
        embeddings = get_embeddings_model()
        vector_store = load_vector_store(embeddings)
    except Exception as e:
        st.error(f"FAISS index not found or failed to load. Error: {e}")
        st.stop()

    # Create retriever
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Define the prompt template
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

    # User query input and processing
    query = st.text_input("Ask a question:")
    if query:
        with st.spinner("Thinking..."):
            answer = rag_chain.invoke(query)
            st.markdown(f"**Answer:** {answer}")

    else:
        # You can also include a fallback response if no query is entered
        st.write("Please ask a question to get a response.")
