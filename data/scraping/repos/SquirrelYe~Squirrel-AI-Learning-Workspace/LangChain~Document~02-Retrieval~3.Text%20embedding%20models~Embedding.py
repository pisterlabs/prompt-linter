from langchain.embeddings import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings()

# 嵌入文档
embeddings = embeddings_model.embed_documents(
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!"
    ]
)
print(len(embeddings))
print(len(embeddings[0]))

# 查询嵌入
query = embeddings_model.embed_query("What was the name mentioned in the conversation?")
print(query[:5])