# from langchain_text_splitters import RecursiveCharacterTextSplitter

# def chunk_text(text, chunk_size=500, chunk_overlap=50):
#     """
#     Splits the text into chunks with a given size and overlap for efficient embedding and retrieval.
#     """
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         length_function=len,
#         is_separator_regex=False,
#     )
#     chunks = text_splitter.create_documents([text])
#     return chunks


from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    """
    Splits the text into chunks with a given size and overlap for efficient embedding and retrieval.
    """
    # Initialize the RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,  # Length function to determine chunk size
        is_separator_regex=False,  # Ensures splitting works on standard characters
    )
    
    # Create documents from the text chunks
    chunks = text_splitter.create_documents([text])
    
    # Return the chunks (Documents)
    return chunks
