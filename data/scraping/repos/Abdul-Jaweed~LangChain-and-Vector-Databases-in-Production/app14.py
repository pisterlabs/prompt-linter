import os
from dotenv import load_dotenv

load_dotenv()

openai_apikey = os.getenv("OPENAI_API_KEY")
cohere_api = os.getenv("COHERE_API_KEY")
activeloop_token = os.getenv("ACTIVELOOP_TOKEN")
os.environ["ACTIVELOOP_TOKEN"] = activeloop_token


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA



# We then create some documents using the RecursiveCharacterTextSplitter class.

texts = [
    "Napoleon Bonaparte was born in 15 August 1769",
    "Louis XIV was born in 5 September 1638",
    "Lady Gaga was born in 28 March 1986",
    "Michael Jeffrey Jordan was born in 17 February 1963"
]


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0
)

docs = text_splitter.create_documents(texts)



# The next step is to create a Deep Lake database and load our documents into it.

# embeddings = OpenAIEmbeddings(model="text-embeddings-ada-002")
embeddings = CohereEmbeddings(
    cohere_api_key=cohere_api,
    model="embed-multilingual-v2.0"
)

my_activeloop_org_id = "abduljaweed"
my_activeloop_dataset_name = "langchain_course_embeddings"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

# adding documents to our DeepLake dataset

db.add_documents(docs)



# We now create a retriever from the database.

retriever = db.as_retriever()


# Finally, we create a RetrievalQA chain in LangChain and run it


# instantiate the LLM Wrapper

model = ChatOpenAI(
    openai_api_key=openai_apikey,
    model="gpt-3.5-turbo"
)

# Create the question-answering chain

qa_chain = RetrievalQA.from_llm(model, retriever=retriever)

# ask a question to the chain

qa_chain.run("When was Michael Jordan born?")