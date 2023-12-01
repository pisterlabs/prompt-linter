from dotenv import load_dotenv

from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import CSVLoader
load_dotenv()
embeddings = OpenAIEmbeddings()

documents = []
# loader = TextLoader('news/summary.txt')
# loader = DirectoryLoader('news', glob="**/*.pdf")
loader = PyPDFLoader('news/Chaos.pdf')
documents.extend(loader.load())
loader = CSVLoader(file_path='news/stocks.csv')  
documents.extend(loader.load())
print(len(documents))
text_splitter = CharacterTextSplitter(chunk_size=2500, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
# print(texts)

docsearch = Chroma.from_documents(texts, embeddings)
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(), 
    chain_type="stuff", 
    retriever=docsearch.as_retriever()
)

def query(q):
    print("Query: ", q)
    print("Answer: ", qa.run(q))

query("Why Chaos testing is impportant for Financial industry?")
query("Who Wrote this article?")
query("How many likes did it get?")