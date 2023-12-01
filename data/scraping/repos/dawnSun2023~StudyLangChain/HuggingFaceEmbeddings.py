from langchain.embeddings import HuggingFaceEmbeddings

#让我们加载 Hugging Face Embedding 类。
embeddings = HuggingFaceEmbeddings()

text = "This is a test document."

query_result = embeddings.embed_query(text)
print(query_result)

doc_result = embeddings.embed_documents([text])
print(doc_result)