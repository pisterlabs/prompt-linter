"""
嵌入（Embeddings）是创建一段表示一段文本的向量。

这很有用，因为它意味着我们可以在向量空间中思考文本，并执行语义搜索之类的操作，在向量空间中查找最相似的文本片段。

LangChain 里有两个方法：向量化一个文档、向量化一个查询。
前者采用多个文本作为输入，而后者采用单个文本。


将它们作为两种单独方法的原因是，某些嵌入提供程序对文档（要搜索的）与查询（搜索查询本身）有不同的嵌入方法。
"""


import env

from langchain.embeddings import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings()

embeddings = embeddings_model.embed_documents(
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!"
    ]
)

print("embeddings: ", embeddings)
print("embeddings length:", len(embeddings))

"""
向量化查询，注意这里仅仅是查询本身向量化了，并没有与上诉的文档向量化进行交互。
"""

embedded_query = embeddings_model.embed_query("What was the name mentioned in the conversation?")
print("query result:", embedded_query[:5])
