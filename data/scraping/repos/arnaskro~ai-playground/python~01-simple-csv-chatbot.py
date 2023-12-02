"""
This is a simple CSV file chatbot that uses the OpenAI API to generate responses.
"""


import dotenv
dotenv.load_dotenv()

# Load CSV data
from langchain.document_loaders import CSVLoader
loader = CSVLoader('../sample-data/01-income-survey.csv')
data = loader.load()

# Split data into chunks

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Create embeddings and vector stores

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

# Load LLM and create a memory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

from langchain.memory import ConversationSummaryMemory
llm = ChatOpenAI()

memory = ConversationSummaryMemory(llm=llm,memory_key="chat_history",return_messages=True)

retriever = vectorstore.as_retriever()
qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)



from colorama import Fore

def loop():
    text = input("\n\n" + Fore.WHITE + ">>> ")

    if text == "exit":
        print (Fore.RED +"Bye!")
    else:
        res = qa(text)
        print(Fore.YELLOW + 'AI:', res['answer'])
        loop()

loop()