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

msg = HumanMessage(content="Tell me about the latest Ant-Man movie. When was it released? What is it about?")

# Call the AI
r = llm(messages=[msg])

# Print the response
print(r.content)

from langchain.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
data_dir = "data/movies"

documents = DirectoryLoader(path=data_dir, glob="*.md", show_progress=True, loader_cls=UnstructuredMarkdownLoader).load()

# Question answering chain
from langchain.chains.question_answering import load_qa_chain

# Prepare the chain and the query
chain = load_qa_chain(llm)
query = "Tell me about the latest Ant Man movie. When was it released? What is it about?"

chain.run(input_documents=documents, question=query)

# Support for callbacks
from langchain.callbacks import get_openai_callback

# Prepare the chain and the query
chain = load_qa_chain(llm, verbose=True)
query = "Tell me about the latest Ant Man movie. When was it released? What is it about?"

# Run the chain, using the callback to capture the number of tokens used
with get_openai_callback() as callback:
    chain.run(input_documents=documents, question=query)
    total_tokens = callback.total_tokens

print(f"Total tokens used: {total_tokens}")