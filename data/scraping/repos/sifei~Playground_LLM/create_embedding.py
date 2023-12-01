from langchain.embeddings import LlamaCppEmbeddings

llama = LlamaCppEmbeddings(model_path="/path/to/model/ggml-model-q4_0.bin")
text = "this is a test sentence."
query_result = llama.embed_query(text)
print(query_result)
