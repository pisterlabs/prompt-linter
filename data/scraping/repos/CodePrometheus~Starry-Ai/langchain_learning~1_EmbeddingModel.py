# ============================ Text Embedding Models ============================ #

from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()  # sentence_transformers
text = "This is a test document."
query_result = embeddings.embed_query(text)
doc_result = embeddings.embed_documents([text])

from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()  # text-embedding-ada-002
text = "This is a test document."
query_result = embeddings.embed_query(text)
doc_result = embeddings.embed_documents([text])
print(query_result)
