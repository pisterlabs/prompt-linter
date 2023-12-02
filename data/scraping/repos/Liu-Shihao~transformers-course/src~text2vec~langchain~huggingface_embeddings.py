from langchain.embeddings import HuggingFaceEmbeddings

"""
默认：https://huggingface.co/sentence-transformers/all-mpnet-base-v2
中文：https://huggingface.co/shibing624/text2vec-base-chinese
"""
embeddings = HuggingFaceEmbeddings(
    model_name="/Users/liushihao/PycharmProjects/model-hub/shibing624/text2vec-base-chinese"
)

text = "This is a test document."

query_result = embeddings.embed_query(text)
print(query_result)

# doc_result = embeddings.embed_documents([text])
# print(doc_result)
