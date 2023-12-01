import constants

import os
import sys

from langchain.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Pinecone
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents import load_tools
from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate, LLMChain, HuggingFaceHub
import pinecone
import prompts

from getpass import getpass

os.environ["OPENAI_API_KEY"] = constants.OPENAI_KEY

if hasattr(constants, 'HUGGINGFACEHUB_KEY'):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = constants.HUGGINGFACEHUB_KEY
if hasattr(constants, 'PINECONE_KEY'):
    os.environ["PINECONE_API_KEY"] = constants.PINECONE_KEY

dir = 'data'
chunks = []

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 512,
    chunk_overlap  = 20,
)

# Go data folder

loader = DirectoryLoader(dir, glob="**/*.txt", loader_cls=TextLoader, silent_errors=True)

# for file in os.listdir(dir):
#   try:
#       # Load up the file as a doc and split
#       loader = PyPDFLoader(os.path.join(dir, file))
#       chunks.extend(loader.load_and_split(text_splitter))
#   except Exception as e:
#       print("Could not load files: ", e)

chunks = loader.load_and_split(text_splitter)

# Create embedding model and llm
repo_id = "tiiuae/falcon-7b"

if "open" in sys.argv:
    llm =HuggingFaceHub(
        repo_id=repo_id, model_kwargs={"temperature": 0.1}
    )
else:
    llm = OpenAI(temperature=0.1)

embeddings = OpenAIEmbeddings()

if os.environ.get("PINECONE_API_KEY"):
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment="us-west4-gcp-free"
    )
    index_name = "langchain1"
    index = pinecone.Index(index_name)

    if "load" in sys.argv:
        db = Pinecone.from_texts([t.page_content for t in chunks], embeddings, index_name=index_name)
    else:
        db = Pinecone(index, embeddings.embed_query, "text")
else:
    # Create vector database and retriever
    db = FAISS.from_documents(chunks, embeddings)

retriever = db.as_retriever(search_kwargs={"k": 5})

# Create paper-searching tool (called tool_paper)
tool_paper = create_retriever_tool(
    retriever,
    name="search_papers",
    description=prompts.DOCSEARCH_DESCRIPTION,
)

# create toolkit (with search tool)
toolkit = [
    tool_paper,
    load_tools(["serpapi"],
               llm=llm,
               serpapi_api_key="fe705be3a03c2fdfb40aa28344a6259a02f35437e0e74fad6c6d93d6e34c71fa")[0]
]
toolkit[1].description = prompts.SEARCH_DESCRIPTION

# writing prompt
prefix = prompts.AGENT_PROMPT_PREFIX
suffix = prompts.AGENT_PROMPT_SUFFIX

# create prompt and memory
prompt = ZeroShotAgent.create_prompt(
    tools=toolkit,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)
memory = ConversationBufferMemory(memory_key="chat_history")

# create agent and agent chain
llm_chain = LLMChain(llm=llm, prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain,
                      tools=toolkit,
                      agent="chat-conversational-react-description",
                      verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=toolkit, verbose=True, memory=memory
)

# instantiate chatbot
print("Welcome to the Transformers chatbot! Type 'exit' to stop.")

while True:
    query = input("Please enter your question: ")

    if query.lower() == 'exit':
        print("Thank you for using the State of the Union chatbot!")
        break

    result = agent_chain({"input": query.strip()})
    print(result["output"])
    print(f'User: {query}')
    print(f'Chatbot: {result["output"]}')

memory.clear()