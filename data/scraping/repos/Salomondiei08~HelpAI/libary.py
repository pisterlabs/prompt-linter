from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import UnstructuredURLLoader, WebBaseLoader
import pinecone
import openai

from utils.secret_keys import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONNEMNT


# Create documents from a list of urls
def createAndSplitTheDocumentFromUrl(urls):
    # Create the document
    loader = WebBaseLoader(urls)
    data = loader.load()

    # Split the document
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(data)
    return docs


# Connect and save a document in pinecone db
def saveDataInPineconeDb(api_key, environment, index_name, docs):
    # initialize pinecone
    pinecone.init(
        api_key=api_key,  # find at app.pinecone.io
        environment=environment,  # next to api key in console
    )

    embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)

    docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    return docsearch


# Connect to pinecode db and use a specific index
def connectToDbAndUseSpecificIndex(api_key, environment, index_name):
    # initialize pinecone
    pinecone.init(
        api_key=api_key,  # find at app.pinecone.io
        environment=environment,  # next to api key in console
    )

    # Set the index
    index = pinecone.Index(index_name)

    # Create the openAI Embedings
    embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)

    # Connect to the vector database
    vectorstore = Pinecone(index, embeddings.embed_query, "text")

    # Return the database
    return vectorstore


# Talk to GPT
def askGPT(text):
    openai.api_key = OPENAI_API_KEY

    # Method for GPT4 (You can use it if you have access to the GPT4 API)
    response = openai.ChatCompletion.create(
   model="gpt-4",
   messages=[
      {"role": "user", "content": text}
   ]
)

    # Method for GPT3.5
#     response = openai.Completion.create(
#    model="gpt-3.5-turbo",
#    messages=[
#       {"role": "user", "content": text}
#    ]
# )
    return print(response.choices[0]["message"]["content"])
