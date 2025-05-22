# from langchain_text_splitters import RecursiveCharacterTextSplitter

# def chunk_text(texts, chunk_size=500, chunk_overlap=50):
#     """
#     Splits one or more strings into chunks for efficient embedding and retrieval.

#     Args:
#         texts (str or list of str): The input text(s) to split.
#         chunk_size (int): The maximum size of each chunk.
#         chunk_overlap (int): The number of characters to overlap between chunks.

#     Returns:
#         list of langchain.schema.Document: The chunked documents.
#     """
#     # input is always a list of texts
#     if isinstance(texts, str):
#         texts = [texts]

#     # Initialize the RecursiveCharacterTextSplitter
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         length_function=len,
#         is_separator_regex=False,  
#     )

#     # Split the texts into documents
#     documents = text_splitter.create_documents(texts)
#     return documents


import nltk
from nltk.tokenize import sent_tokenize
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util

nltk.download("punkt")


class ChunkerFactory:
    def __init__(
        self,
        method: str = "recursive",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        semantic_threshold: float = 0.7
    ):
        self.method = method
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.semantic_threshold = semantic_threshold

        # Load models conditionally
        if method == "semantic":
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        elif method == "sentence":
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def chunk(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        if self.method == "recursive":
            return self._recursive_chunk(texts)
        elif self.method == "sentence":
            return self._sentence_chunk(texts)
        elif self.method == "paragraph":
            return self._paragraph_chunk(texts)
        elif self.method == "semantic":
            return self._semantic_chunk(texts)
        else:
            raise ValueError(f"Unknown chunking method: {self.method}")

    def _recursive_chunk(self, texts):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        return splitter.create_documents(texts)

    def _sentence_chunk(self, texts):
        documents = []
        for text in texts:
            sentences = sent_tokenize(text)
            current_chunk = ""

            for sentence in sentences:
                temp_chunk = current_chunk + " " + sentence
                if len(self.tokenizer.encode(temp_chunk)) <= self.chunk_size:
                    current_chunk = temp_chunk
                else:
                    if current_chunk.strip():
                        documents.append(Document(page_content=current_chunk.strip()))
                    current_chunk = sentence
            if current_chunk.strip():
                documents.append(Document(page_content=current_chunk.strip()))
        return documents

    def _paragraph_chunk(self, texts):
        documents = []
        for text in texts:
            paragraphs = text.split("\n\n")
            for para in paragraphs:
                if para.strip():
                    documents.append(Document(page_content=para.strip()))
        return documents

    def _semantic_chunk(self, texts):
        documents = []
        for text in texts:
            sentences = sent_tokenize(text)
            if not sentences:
                continue

            embeddings = self.embedding_model.encode(sentences)
            current_chunk = [sentences[0]]

            for i in range(1, len(sentences)):
                sim = util.cos_sim(embeddings[i], embeddings[i - 1]).item()
                if sim >= self.semantic_threshold:
                    current_chunk.append(sentences[i])
                else:
                    documents.append(Document(page_content=" ".join(current_chunk)))
                    current_chunk = [sentences[i]]

            if current_chunk:
                documents.append(Document(page_content=" ".join(current_chunk)))

        return documents
