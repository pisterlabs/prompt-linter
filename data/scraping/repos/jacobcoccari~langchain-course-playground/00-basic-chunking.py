from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter


loader = TextLoader(
    "./09-chunking-embeddings-and-vector-stores/jfk-inaguration-speech.txt"
)
data = loader.load()

speech = data[0].page_content

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)


texts = text_splitter.create_documents([speech])
# print(texts)

# We can also pass custom metadata about each of the document, in case it might be helpful to us downstream.

metadatas = [{"title": "JFK Inauguration Speech", "author": "John F. Kennedy"}]

texts_with_metadata = text_splitter.create_documents([speech], metadatas=metadatas)
print(texts_with_metadata[0])
print(len(texts_with_metadata))
