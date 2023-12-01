# https://github.com/gkamradt/langchain-tutorials/blob/main/data_generation/Custom%20Files%20Question%20%26%20Answer.ipynb

"""
Custom Files Question & Answer. 文档问答包括四个步骤：
1. 创建索引
2. 从该索引创建一个检索器
3. 创建问答链
4. 问问题！
"""

# pip install chromadb
from langchain.chains import VectorDBQA
from langchain.document_loaders import DirectoryLoader
from langchain.llms import OpenAI

# 加载文件夹中的所有txt类型的文件，并转成 document 对象
loader = DirectoryLoader('./data/', glob='**/*.txt')
documents = loader.load()
# 接下来，我们将文档拆分成块。
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
# 然后我们将选择我们想要使用的嵌入。
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
# 我们现在创建 vectorstore 用作索引，并进行持久化
from langchain.vectorstores import Chroma

# vector_store = Chroma.from_documents(texts, embeddings, persist_directory="./vector_store")
# vector_store.persist()
vector_store = Chroma(persist_directory="./vector_store", embedding_function=embeddings)
# 创建一个链并用它来回答问题！
qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=vector_store,
                                return_source_documents=True)
print(qa.input_keys, qa.output_keys)

res = qa({"query": "出差申请单修改"})
print(res['result'])
print(res['source_documents'])
