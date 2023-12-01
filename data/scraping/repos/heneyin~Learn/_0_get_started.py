from langchain.document_loaders import TextLoader

loader = TextLoader("./_0_介绍.py")
loaderResult = loader.load()
# 输出包含内容与元数据
print("loaderResult", loaderResult)



