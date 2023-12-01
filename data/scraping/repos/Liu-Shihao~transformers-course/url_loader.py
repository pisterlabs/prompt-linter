from langchain.document_loaders import UnstructuredURLLoader
"""
pip install unstructured
pip install tabulate
pip install pdf2image
pip install libmagic
"""
urls = [
    "https://baike.baidu.com/item/embedding/52797850?fr=aladdin",
    "https://python.langchain.com/en/latest/modules/indexes/document_loaders.html"
]
loader = UnstructuredURLLoader(urls=urls)

data = loader.load()
print(data)