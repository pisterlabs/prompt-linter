# dotenv is a library that allows us to securely load env variables
from dotenv import load_dotenv 

# used to load an individual file (TextLoader) or multiple files (DirectoryLoader)
from langchain.document_loaders import TextLoader, DirectoryLoader

# used to split the text within documents and chunk the data
from langchain.text_splitter import CharacterTextSplitter

# use embedding from OpenAI (but others available)
from langchain.embeddings import OpenAIEmbeddings

# using Chroma database to store our vector embeddings
from langchain.vectorstores import Chroma

# use this to configure the Chroma database  
from chromadb.config import Settings

# We'll use the chain that allows Question and Answering and provides source of where it got the data from. This is useful if you have multiple files. If you don't need the source, you can use RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain

# We'll use the OpenAI Chat model to interact with the embeddings. This is the model that allows us to query in a similar way to ChatGPT
from langchain.chat_models import ChatOpenAI

# We'll need this for reading/storing from directories
import os

# looks for the .env file and loads the variable(s) 
load_dotenv()

# prepare directories for DB

ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DB_DIR: str = os.path.join(ABS_PATH, "db")

# use TextLoader for an individual file
# explicitly stating the encoding is also recommmended
doc_loader: TextLoader = TextLoader('MSFT_Call_Transcript.txt', encoding="utf8")

# if you want to load multiple files, place them in a directory 
# and use DirectoryLoader; comment above and uncomment below
#doc_loader: DirectoryLoader = DirectoryLoader('my_directory')

# load the document
document: str = doc_loader.load()

# obtain an instance of the splitter with the relevant parameters 
text_splitter: CharacterTextSplitter = CharacterTextSplitter(chunk_size=512 , chunk_overlap=0)

# split the document data
split_docs: list[str] = text_splitter.split_documents(document)

# load the embeddings from OpenAI
openai_embeddings: OpenAIEmbeddings = OpenAIEmbeddings()


# configure our database
client_settings: Settings = Settings(
    chroma_db_impl="duckdb+parquet", #we'll store as parquet files/DuckDB
    persist_directory=DB_DIR, #location to store 
    anonymized_telemetry=False # optional but showing how to toggle telemetry
)

# check if the database exists already
# if not, create it, otherwise read from the database
if not os.path.exists(DB_DIR):
    # Create the database from the document(s) above and use the OpenAI embeddings for the word to vector conversions. We also pass the "persist_directory" parameter which means 
    # this won't be a transient database, it will be stored on the hard drive at the DB_DIR location. We also pass the settings we created earlier and give the collection a name
    vector_store: Chroma = Chroma.from_documents(split_docs, openai_embeddings,  persist_directory=DB_DIR, 
                                         client_settings=client_settings,
                                         collection_name="transcripts_store")

    # It's key to called the persist() method otherwise it won't be saved 
    vector_store.persist()
else:
    # As the database already exists, load the collection from there
    vector_store: Chroma = Chroma(collection_name="transcripts_store", persist_directory=DB_DIR, embedding_function=openai_embeddings, client_settings=client_settings)


# create and configure our chain
# we're using ChatOpenAI LLM with the 'gpt-3.5-turbo' model
# we're setting the temperature to 0. The higher the temperature, the more 'creative' the answers. In my case, I want as factual and direct from source info as possible
# 'stuff' is the default chain_type which means it uses all the data from the document
# set the retriever to be our embeddings database
qa_with_source: RetrievalQAWithSourcesChain = RetrievalQAWithSourcesChain.from_chain_type(
     llm=ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo'),
     chain_type="stuff",     
     retriever = vector_store.as_retriever()
    )

def query_document(question: str) -> dict[str, str]:
    return qa_with_source({"question": question})
    
# loop through to allow the user to ask questions until they type in 'quit'
while(True):
    # make the user input yellow using ANSI codes
    print("What is your query? ", end="")
    user_query : str = input("\033[33m")
    print("\033[0m")
    if(user_query == "quit"):
        break
    response: dict[str, str] = query_document(user_query)
    # make the answer green and source blue using ANSI codes
    print(f'Answer: \033[32m{response["answer"]}\033[0m')
    print(f'\033[34mSources: {response["sources"]}\033[0m')



