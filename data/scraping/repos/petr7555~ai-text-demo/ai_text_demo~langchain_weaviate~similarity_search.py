from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Weaviate

from ai_text_demo.constants import WEAVIATE_URL, OPENAI_API_KEY
from ai_text_demo.utils import relative_path_from_file

DATA_PATH = relative_path_from_file(__file__, "data/state_of_the_union.txt")

loader = TextLoader(DATA_PATH)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

db = Weaviate.from_documents(docs, embeddings, weaviate_url=WEAVIATE_URL, by_text=False)

print("Similarity search:")
query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)

print(docs[0].page_content)

print("\nSimilarity search with score:")

docs = db.similarity_search_with_score(query, by_text=False)
print(docs[0])

print("\nMMR:")
retriever = db.as_retriever(search_type="mmr")
print(retriever.get_relevant_documents(query)[0])
