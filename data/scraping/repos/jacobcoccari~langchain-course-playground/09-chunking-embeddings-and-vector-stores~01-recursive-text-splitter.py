from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


loader = TextLoader(
    "./09-chunking-embeddings-and-vector-stores/jfk-inaguration-speech.txt"
)
data = loader.load()

speech = data[0].page_content

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    separators=["\n\n", "\n", ".", " "],  # default list
    chunk_size=500,
    chunk_overlap=50,
    # length_function=len,
)

texts = text_splitter.create_documents([speech])
print(texts[0])
print(len(texts))
