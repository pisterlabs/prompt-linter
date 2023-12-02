from langchain.document_loaders import GutenbergLoader
#小说阅读器
loader = GutenbergLoader("https://www.gutenberg.org/cache/epub/16389/pg16389.txt")

data = loader.load()

print(data[0].page_content)