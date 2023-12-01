from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains import LLMChain


# Load dummy documents
from langchain.document_loaders import TextLoader
loader = TextLoader("data/morning_news.txt")
documents = loader.load()

# Init openai keys
from dotenv import load_dotenv

load_dotenv()

# Init pinecone
import pinecone

PINECONE_API_KEY = "df34ba21-9d12-47f8-99a6-a70752693823"
PINECONE_ENV = "us-west1-gcp-free"

# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_ENV,  # next to api key in console
)
index_name = "langchain-demo"

# Check if vectorstore exists
if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, metric="cosine", shards=1, dimension=1536)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
# print(f"Split {len(documents)} documents into {len(docs)} chunks")
# Store 
embeddings = OpenAIEmbeddings()
# print(f'Model loaded: {dir(embeddings.model)}')
vectorstore = Pinecone.from_documents(docs, embeddings, index_name=index_name)

llm = OpenAI(temperature=0)
question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
doc_chain = load_qa_with_sources_chain(llm, chain_type="map_reduce")

chain = ConversationalRetrievalChain(
    retriever=vectorstore.as_retriever(),
    question_generator=question_generator,
    combine_docs_chain=doc_chain,
)

history = []
while True:
    query = input("Enter a question: ")
    result = chain({"question": query, "chat_history": []})
    history.append(query)
    history.append(result["answer"])
    print(result["answer"])



