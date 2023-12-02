from langchain.document_loaders import TextLoader

loader = TextLoader("./08-Data-Connections/jfk-inaguration-speech.txt")
data = loader.load()

print(data)
print(data[0].page_content)
