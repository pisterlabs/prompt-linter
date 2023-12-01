from langchain.document_loaders import BiliBiliLoader

loader = BiliBiliLoader(["https://www.bilibili.com/video/BV1nE41117BQ/?p=130"])

str = loader.load()
print(str)