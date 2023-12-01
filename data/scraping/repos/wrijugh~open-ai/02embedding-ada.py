import os
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv

# Load environment variables
if load_dotenv(dotenv_path="../.env"):
    print("Found OpenAPI Base Endpoint: " + os.getenv("OPENAI_API_BASE"))
else: 
    print("No file .env found")

openai_api_type = os.getenv("OPENAI_API_TYPE")
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_base = os.getenv("OPENAI_API_BASE")
openai_api_version = os.getenv("OPENAI_API_VERSION")
deployment_name = os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME")
embedding_name = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

# Create an instance of Azure OpenAI
llm = AzureChatOpenAI(
    openai_api_type = openai_api_type,
    openai_api_version = openai_api_version,
    openai_api_base = openai_api_base,
    openai_api_key = openai_api_key,
    deployment_name = deployment_name,
    temperature = 0
)

from langchain.embeddings import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings(
    openai_api_type = openai_api_type,
    openai_api_version = openai_api_version,
    openai_api_base = openai_api_base,
    openai_api_key = openai_api_key,
    deployment = embedding_name,
    chunk_size = 1
)

from langchain.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
data_dir = "data/movies"
documents = DirectoryLoader(path=data_dir, glob="*.md", show_progress=True, loader_cls=UnstructuredMarkdownLoader).load()

from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
document_chunks = text_splitter.split_documents(documents)

from langchain.vectorstores import Qdrant

qdrant = Qdrant.from_documents(
    document_chunks,
    embeddings_model,
    location=":memory:",
    collection_name="movies",
)

retriever = qdrant.as_retriever()

from langchain.chains import RetrievalQA
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

query = "Tell me about the latest MCU movie. When was it released? What is it about?"
qa.run(query)

from langchain.callbacks import get_openai_callback
with get_openai_callback() as callback:
    qa.run(query)
    total_tokens = callback.total_tokens

print(f"Total tokens used: {total_tokens}")

