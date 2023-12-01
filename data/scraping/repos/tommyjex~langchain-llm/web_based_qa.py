from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://zhuanlan.zhihu.com/p/597586623")
data = loader.load()
print(data)