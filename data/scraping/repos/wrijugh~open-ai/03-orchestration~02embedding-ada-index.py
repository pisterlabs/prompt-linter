import os
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
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

#THis will USE less TOKENS

from langchain.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
data_dir = "data/movies"

from langchain.indexes import VectorstoreIndexCreator
loader = DirectoryLoader(path=data_dir, glob="*.md", show_progress=True, loader_cls=UnstructuredMarkdownLoader)

index = VectorstoreIndexCreator(
    embedding=embeddings_model
).from_loaders([loader])

query = "Tell me about the latest MCU movie. When was it released? What is it about?"
index.query(llm=llm, question=query)

from langchain.callbacks import get_openai_callback
with get_openai_callback() as callback:
    index.query(llm=llm, question=query)
    total_tokens = callback.total_tokens

print(f"Total tokens used: {total_tokens}")
