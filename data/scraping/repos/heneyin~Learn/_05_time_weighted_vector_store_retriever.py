"""
https://python.langchain.com/docs/modules/data_connection/retrievers/how_to/time_weighted_vectorstore

该检索器使用语义相似性和时间衰减的组合。

对它们进行评分的算法是：

```
semantic_similarity + (1.0 - decay_rate) ^ hours_passed
```

值得注意的是，hours_passed 指的是自上次访问检索器中的对象以来经过的小时数，而不是自创建以来经过的小时数。这意味着经常访问的对象保持“新鲜”。

"""
import env

import faiss

from datetime import datetime, timedelta
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import Document
from langchain.vectorstores import FAISS

# Define your embedding model
embeddings_model = OpenAIEmbeddings()
# Initialize the vectorstore as empty
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

"""
低衰减率
# 衰减率为 0 意味着记忆永远不会被遗忘，使得该检索器相当于向量查找。
"""

retriever = TimeWeightedVectorStoreRetriever(vectorstore=vectorstore, decay_rate=.0000000000000000000000001, k=1)

yesterday = datetime.now() - timedelta(days=2)
# 设置了最后访问时间为昨天
retriever.add_documents([Document(page_content="hello world 1", metadata={"last_accessed_at": yesterday})])
retriever.add_documents([Document(page_content="hello world 2", metadata={"last_accessed_at": datetime.now()})])

# "Hello World" is returned first because it is most salient, and the decay rate is close to 0., meaning it's still recent enough
result = retriever.get_relevant_documents("hello world")
# 显示最近出现的文档。
print("低衰减率 get_relevant_documents:", result)

"""
高衰减率
"""

# Define your embedding model
embeddings_model = OpenAIEmbeddings()
# Initialize the vectorstore as empty
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
retriever = TimeWeightedVectorStoreRetriever(vectorstore=vectorstore, decay_rate=.999, k=1)

yesterday = datetime.now() - timedelta(days=1)
retriever.add_documents([Document(page_content="hello world 1", metadata={"last_accessed_at": yesterday})])
retriever.add_documents([Document(page_content="hello world 2", metadata={"last_accessed_at": datetime.now()})])

# "Hello Foo" is returned first because "hello world" is mostly forgotten
result = retriever.get_relevant_documents("hello world")

print("高衰减率 get_relevant_documents:", result)