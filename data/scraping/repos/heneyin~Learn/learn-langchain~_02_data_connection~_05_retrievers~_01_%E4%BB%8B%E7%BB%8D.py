"""
retrievers : 检索器

检索器是一个接口，它根据非结构化查询返回文档。

它比向量存储更通用。

检索器不需要能够存储文档，只需返回（或检索）它。

向量存储可以用作检索器的骨干，但也有其他类型的检索器。
"""


"""
您可以调用 get_relevant_documents 或异步 get_relevant_documents 方法来检索与查询相关的文档，
其中“相关性”由您调用的特定检索器对象定义。
"""

"""
我们关注的主要检索器类型是 Vectorstore 检索器。我们将在本指南的其余部分重点讨论这一点
langchain 默认使用 Chroma 作为向量数据库
"""

"""
针对文档的问答包括四个步骤：
1. 创建一个 index
2. 从 index 创建一个检索器
3. 创建一个问题与回答的chain
4. 回答提问。
"""

import env

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# 使用 loader 加载文档
from langchain.document_loaders import TextLoader
loader = TextLoader('../../texts/maodun.txt', encoding='utf8')

# 一行代码。根据 loader 创建 index
from langchain.indexes import VectorstoreIndexCreator
index = VectorstoreIndexCreator().from_loaders([loader])

llm = OpenAI(max_tokens=8000,
             model_name="gpt-3.5-turbo-16k-0613")

# 现在索引已创建，我们可以使用它来询问数据问题！请注意，在幕后，这实际上也执行了几个步骤，我们将在本指南的后面部分介绍这些步骤
query = "马关条约赔偿了多少白银"
# 只获得查询结果
queryResult = index.query(query, llm=llm)
print("queryResult", queryResult)

# 获得查询结果与源文档
query = "马关条约赔偿了多少白银"
queryResult = index.query_with_sources(query, llm=llm)
print("queryResult with sources", queryResult)
