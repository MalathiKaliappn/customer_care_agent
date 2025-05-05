from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def build_chain(model, retriever):
    template = """
<|system|>
Answer the question based on the given context. If you don't know the answer, say "I don't know".

{context}

{question}
"""
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    llm_chain = prompt | model | StrOutputParser()
    return {"context": retriever, "question": RunnablePassthrough()} | llm_chain
