from langchain.text_splitter import TokenTextSplitter

from langchain.document_loaders import TextLoader

loader = TextLoader("data/ai.txt")

documents = loader.load()

text_splitter = TokenTextSplitter(
  separator="\n\n",
  chunk_size=30,
  chunk_overlap=5
)

texts = text_splitter.split_documents(documents)

print(texts[0].page_content)
print(texts[1].page_content)




