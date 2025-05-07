from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(texts, chunk_size=500, chunk_overlap=50):
    """
    Splits one or more strings into chunks for efficient embedding and retrieval.

    Args:
        texts (str or list of str): The input text(s) to split.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        list of langchain.schema.Document: The chunked documents.
    """
    # input is always a list of texts
    if isinstance(texts, str):
        texts = [texts]

    # Initialize the RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,  
    )

    # Split the texts into documents
    documents = text_splitter.create_documents(texts)
    return documents
