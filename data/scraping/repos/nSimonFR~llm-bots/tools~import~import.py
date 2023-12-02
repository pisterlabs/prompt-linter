from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
import pinecone

loader = TextLoader('./state_of_the_union.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

pinecone.init(
    api_key=os.environ.get('PINECONE_API_KEY'),
    environment=os.environ.get('PINECONE_ENV'),
)
index_name = os.environ.get('PINECONE_INDEX')

embeddings = OpenAIEmbeddings()
docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)

query = "What did the president say about Ketanji Brown Jackson"
docs = docsearch.similarity_search(query)
