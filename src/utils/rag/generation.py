# from langchain.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough

# def build_chain(model, retriever):
#     template = """
# <|system|>
# Answer the question based on the given context. If you don't know the answer, say "I don't know".

# {context}

# {question}
# """
#     prompt = PromptTemplate(input_variables=["context", "question"], template=template)
#     llm_chain = prompt | model | StrOutputParser()
#     return {"context": retriever, "question": RunnablePassthrough()} | llm_chain

from langchain.prompts import PromptTemplate
from langchain.output_parsers import StrOutputParser
from langchain.runnables import RunnablePassthrough

def build_chain(model, retriever):
    # Define the prompt template
    template = """
    <|system|>
    Answer the question based on the given context. If you don't know the answer, say "I don't know".
    
    {context}
    
    {question}
    """
    
    # Create the prompt template
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    
    # Build the LLM chain with the prompt, model, and output parser
    llm_chain = prompt | model | StrOutputParser()
    
    # Return the final chain that first retrieves context from the retriever and then processes the question
    return {"context": retriever, "question": RunnablePassthrough()} | llm_chain
