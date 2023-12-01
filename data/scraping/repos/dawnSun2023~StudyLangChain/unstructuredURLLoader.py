from langchain.document_loaders import UnstructuredURLLoader
urls = [
    "https://www.baidu.com"
]
loader = UnstructuredURLLoader(urls=urls)

data = loader.load()
print(data)