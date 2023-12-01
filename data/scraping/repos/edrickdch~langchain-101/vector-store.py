from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


with open("example_data/state_of_the_union.txt") as f:
    state_of_the_union = f.read()

text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
)
texts = text_splitter.create_documents([state_of_the_union])

embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_texts(texts, embeddings)

query = "What did the president say about Ketanji Brown Jackson"
docs = docsearch.similarity_search(query)


print(docs[0].page_content)
