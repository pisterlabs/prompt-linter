
import os
import openai
from dotenv import load_dotenv

from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import OpenAI
from langchain.llms import AzureOpenAI

load_dotenv()
# Configure OpenAI API
openai.api_type = "azure"
openai.api_version = "2022-12-01"
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv("OPENAI_API_KEY")

# Create a completion - qna를 위하여 davinci 모델생성
llm = AzureOpenAI(deployment_name="text-davinci-003")

# text embedding 을 위해서는 좀더 저렴한 ada 모델을 사용
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=1)
# embeddings = OpenAIEmbeddings()

# loader = TextLoader('news/summary.txt')
loader = DirectoryLoader('mydata', glob="**/*.txt")

documents = loader.load()
# print(len(documents))
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
print(texts)

docsearch = Chroma.from_documents(texts, embeddings)
qa = RetrievalQA.from_chain_type(
    # llm=OpenAI(), 
    llm=llm, 
    chain_type="stuff", 
    # retriever=docsearch.as_retriever()
    retriever=docsearch.as_retriever(search_kwargs={"k": 1})
)

def query(q):
    print("Query: ", q)
    print("Answer: ", qa.run(q))

# query("퓨어스토리지의 플래시블레이드는 무엇인가?")
# query("퓨어스토리지의 플래시블레이드는 경제성이 좋은가?")
query("퓨어스토리지의 플래시블레이드는 기업에 적합한가?")
# query("What are the effects of legislations surrounding emissions on the Australia coal market?")
# query("What are China's plans with renewable energy?")
# query("Is there an export ban on Coal in Indonesia? Why?")
# query("Who are the main exporters of Coal to China? What is the role of Indonesia in this?")