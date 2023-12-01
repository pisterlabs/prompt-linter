from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

text = "This is a test document."

query_result = embeddings.embed_query(text)
print(query_result)

doc_result = embeddings.embed_documents([text])
print(doc_result)
