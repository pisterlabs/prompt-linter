"""
本脚本旨在从特定网页执行文档检索和问答操作。流程包括以下步骤：

1. 使用WebBaseLoader类从指定URL加载文档。
2. 使用RecursiveCharacterTextSplitter类将文档切割成更小的文本块。
3. 使用OpenAIEmbeddings类为这些文本块创建文本嵌入。
4. 使用Chroma类与这些嵌入建立向量存储库。
5. 利用RetrievalQA类结合OpenAI语言模型构建基于检索的问答系统。

提供的URL是Lilian Weng关于'agent'主题的博客文章。脚本随后将博客文章切分成可管理的文本块，为这些块创建嵌入，并构建一个向量存储库。最后，它设置了一个基于检索的问答系统，查询用以解释'任务分解'的概念。

注意：RetrievalQA.from_chain_type方法中的chain_type参数设置为占位值"stuff"。需要用针对具体应用或上下文的有效参数替换它，以确保正确运作。
"""

# 导入WebBaseLoader类，用于从网页加载文档
from langchain.document_loaders import WebBaseLoader
# 导入RecursiveCharacterTextSplitter类，用于将文档切割成较小的文本块
from langchain.text_splitter import RecursiveCharacterTextSplitter
# 导入OpenAIEmbeddings类，用于创建文本嵌入
from langchain.embeddings import OpenAIEmbeddings
# 导入Chroma类，用于构建向量存储库
from langchain.vectorstores import Chroma
# 导入RetrievalQA类和OpenAI类，用于构建检索式问答系统
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# 初始化WebBaseLoader，将指定的URL作为参数
# 这个URL是Lilian Weng的博客，关于"agent"的一个帖子
loader = WebBaseLoader("https://zhuanlan.zhihu.com/p/74994510")
# 使用loader加载文档，这里的文档是指上面URL中的内容
documents = loader.load()

# 初始化文本切割器，设置每个文本块大小为500字符，并且文本块之间没有重叠
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
# 将加载的文档切割成文本块
texts = text_splitter.split_documents(documents)

# 创建OpenAIEmbeddings实例，这将使用OpenAI模型将文本转换成向量
embeddings = OpenAIEmbeddings()

# 使用分割好的文本块和嵌入向量构建向量存储库
# 这个库将用于后续的文档搜索和信息检索
docsearch = Chroma.from_documents(texts, embeddings)

# 创建检索式问答链（RetrievalQA），使用OpenAI模型和先前构建的向量存储库
# "chain_type"参数在这里被设置为"stuff"，但它是一个示例值，通常应根据实际需求设置
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())

# 定义一个查询问题："详细告诉我什么是任务分解？"
query = "什么是大数据分析?"
# 执行查询，qa系统会在文档中检索信息并生成答案
answer = qa.run(query)
print(answer)
