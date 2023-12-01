import os
import openai
import langchain
from tenacity import retry, stop_after_attempt, wait_fixed, wait_exponential, wait_random_exponential
from PyPDF2 import PdfReader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from pathlib import Path
import dotenv


env_name = os.environ["APP_ENV"] if "APP_ENV" in os.environ else "local"

# Load env settings
env_file_path = Path(f"./.env.{env_name}")
print(f"Loading environment from: {env_file_path}")
with open(env_file_path) as f:
    dotenv.load_dotenv(dotenv_path=env_file_path)
# print(os.environ)

openai.api_type: str = "azure"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
model: str = os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME")

# openai_client = openai.AzureOpenAI(
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
# )
embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
    azure_deployment=model,
    model=model,
    chunk_size=1,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_type="azure",
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"]
)

langchain.verbose = False

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 10


def get_pdf_text(files):
    text = ""
    for f in files:
        print(f"Processing {f.name}")
        # print(f"f object: {dir(f)}")
        reader = PdfReader(f)
        page_count = len(reader.pages)
        for page in reader.pages:
            text += page.extract_text()
        print(f"Page count: {page_count}")

    return text


def load_csv(csv_file):
    csv_loader = CSVLoader(file_path=csv_file)
    return csv_loader.load()


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    print(f"Number of chunks: {len(chunks)}")
    return chunks


def get_vectors(chunks):

    global embeddings

    if not embeddings:
        print("Embeddings not initialized. Initializing now.")
        return
    # text_embeddings = embeddings.embed_documents(chunks)
    # text_embeddings = [vectorize_with_delay(
    #     embeddings, chunk) for chunk in chunks]
    vector_store = FAISS.from_texts(chunks, embeddings)

    # text_embeddings = embed_with_delay(embeddings, chunks)
    # text_embedding_pairs = zip(chunks, text_embeddings)
    # text_embedding_pairs_list = list(text_embedding_pairs)
    # vector_store = FAISS.from_embeddings(text_embedding_pairs_list, embeddings)

    return vector_store


@retry(wait=wait_random_exponential(multiplier=1, min=4, max=10))
def vectorize_with_delay(embeddings, document):
    return embeddings.vectorize(document)


@retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(6))
def embed_with_delay(embeddings, document):
    return embeddings.embed_documents(document)


def get_conversation_chain(vector_store):

    llm = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_CHATGPT_DEPLOYMENT_NAME"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        openai_api_type="azure",
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        openai_api_key=os.environ["AZURE_OPENAI_API_KEY"]
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )

    return conversation_chain
