from langchain.chat_models import ChatOpenAI # ChatOpenAI enables you to chat with GPT-3
from langchain.chains import RetrievalQA # RetrievalQA enables you to retrieve answers from a vector store
from langchain.document_loaders import GoogleDriveLoader # GoogleDriveLoader enables you to load documents from Google Drive
from langchain.embeddings import OpenAIEmbeddings # OpenAIEmbeddings enables you to embed text with GPT-3 ie convert text to vectors
from langchain.vectorstores import Chroma # Chroma enables you to store vectors
from langchain.text_splitter import RecursiveCharacterTextSplitter # RecursiveCharacterTextSplitter enables you to split text into chunks
import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

folder_id = '1K7qXSMy_SKkug3ZX5DN-2dkkrKPUkPH8'
loader = GoogleDriveLoader(folder_id = folder_id, 
                           recursive=False)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000, chunk_overlap=20, separators=[" ", ",", "\n"]
)

texts = text_splitter.split_documents(docs)
#embeddings = OpenAIEmbeddings(model="davinci")

#persist_directory = "gpt_drive"
#metadata = {"folder_id": folder_id}
db = Chroma.from_documents(texts, embedding=OpenAIEmbeddings(model="davinci"), collection_name='annualreports')
# collection_name helps you identify the vector store and is used by the RetrievalQA class
retriever = db.as_retriever()

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

while True:
    question = input("> ")
    answer = qa.run(question)
    print(answer)