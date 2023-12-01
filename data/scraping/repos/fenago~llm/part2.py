from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import OpenAI

load_dotenv()
embeddings = OpenAIEmbeddings()

loader = DirectoryLoader('data', glob="**/*.txt")

documents = loader.load()
print(len(documents))
text_splitter = CharacterTextSplitter(chunk_size=2500, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

docsearch = Chroma.from_documents(texts, embeddings)
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(), 
    chain_type="stuff", 
    retriever=docsearch.as_retriever()
)

def query(q):
    print("Query: ", q)
    print("Answer: ", qa.run(q))

query("What is machine learning?")
query("How do you apply a linear transformation?")
query("How do you load data? What does the code look like for this?")
query("What are the different types of data that you can load?")
query("What is the flight path of the Monarch Butterfly?")
