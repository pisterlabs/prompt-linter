## LENRAutoGPT v0.3
## Added the following:
#1. Weaviate client
#2. User/Agent prompt communications (work in progress)
#3. Added logging information

import os
import logging
from dotenv import load_dotenv
import requests
from PyPDF2 import PdfFileReader
from io import BytesIO
from weaviate import Client

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the environment variables from .env file
load_dotenv()

# Get the API keys from environment variables
serpapikey = os.getenv('SERPAPI_API_KEY')
openaikey = os.getenv('OPENAI_API_KEY')
wolframalphaapikey = os.getenv('WOLFRAM_ALPHA_APPID')
googleapikey = os.getenv('GOOGLE_API_KEY')
googlesearch = os.getenv('GOOGLE_CSE_ID')
weaviateapikey = os.getenv('WEAVIATE_API_KEY')
weaviateurl = os.getenv('WEAVIATE_URL')

# Import your modules
from langchain.utilities.serpapi import SerpAPIWrapper
from langchain.utilities.arxiv import ArxivAPIWrapper
from langchain.utilities.wikipedia import WikipediaAPIWrapper
from langchain.utilities.google_search import GoogleSearchAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
import faiss

# Define the ReadAndStoreLocalPDFTool
class ReadAndStoreLocalPDFTool:
    def __init__(self, docstore, chunk_size=500):
        self.name = "read_and_store_local_pdf"
        self.description = "Useful for reading a PDF from local filesystem and storing its content."
        self.docstore = docstore
        self.chunk_size = chunk_size

    def run(self, filepath):
        try:
            with open(filepath, "rb") as file:
                pdf = PdfFileReader(file)
                num_pages = pdf.getNumPages()
                chunks = []
                for page in range(num_pages):
                    text = pdf.getPage(page).extractText()
                    chunks.extend(self._split_text_to_chunks(text))
            # Store the chunks to docstore
            self.docstore[filepath] = chunks
            return chunks
        except FileNotFoundError:
            logging.error(f"The file {filepath} was not found.")
            return None
        except PdfReadError:
            logging.error(f"The file {filepath} is not a valid PDF.")
            return None

    def _split_text_to_chunks(self, text):
        chunks = []
        words = text.split()
        current_chunk = ""
        for word in words:
            if len(current_chunk) + len(word) + 1 > self.chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            current_chunk += word + " "
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

# Define your tools
wolframalpha = WolframAlphaAPIWrapper()
wikipedia = WikipediaAPIWrapper()
googlesearch = GoogleSearchAPIWrapper()
arxiv = ArxivAPIWrapper()
read_and_store_pdf_tool = ReadAndStoreLocalPDFTool(InMemoryDocstore({}), chunk_size=500)

tools = [
    Tool(
        name="wikisearch",
        func=wikipedia.run,
        description="Useful for searching the web for information."
    ),
    Tool(
        name="googlesearch",
        func=googlesearch.run,
        description="Useful for searching the web for information."
    ),
    Tool(
        name="arxiv",
        func=arxiv.run,
        description="Useful for when you need scientific research papers."
    ),
    Tool(
        name="wolframalpha",
        func=wolframalpha.run,
        description="Useful for answering factual questions."
    ),
    WriteFileTool(),
    ReadFileTool(),
    Tool(
        name="read_and_store_local_pdf",
        func=read_and_store_pdf_tool.run,
        description="Useful for reading a PDF from local filesystem and storing its content."
    ),
]

# Define your embedding model
embeddings_model = OpenAIEmbeddings()

# Initialize the Weaviate client
weaviate_url = "WEAVIATE_URL"
weaviate_client = Client(weaviate_url, weaviateapikey)

# Define your vector store
class WeaviateVectorStore:
    def __init__(self, weaviate_client):
        self.client = weaviate_client

    def add_vector(self, id, vector):
        self.client.data_object.create({"vector": vector}, id)

    def get_vector(self, id):
        return self.client.data_object.get(id)["vector"]

# Initialize the vectorstore as empty FAISS index
embedding_size = 1536  # openai embeddings has 1536 dimensions
index = faiss.IndexFlatL2(embedding_size)  # Index that stores the full vectors and performs exhaustive search

# Initialize the vector store
weaviate_vectorstore = WeaviateVectorStore(weaviate_client)

class FAISS:
    def __init__(self, embedding_func, index, docstore, weaviate_store, initial_data=None):
        self.embedding_func = embedding_func
        self.index = index
        self.docstore = docstore
        self.weaviate_store = weaviate_store
        if initial_data:
            self.add_batch(list(initial_data.keys()), list(initial_data.values()))

    def add_batch(self, keys, texts):
        vectors = self.embedding_func(texts)
        assert len(vectors) == len(keys)
        for key, vector in zip(keys, vectors):
            self.index.add(np.array([vector]))  # Add to FAISS index
            self.docstore[key] = vector  # Add to docstore
            self.weaviate_store.add_vector(key, vector.tolist())  # Add to Weaviate

    def as_retriever(self):
        class Retriever:
            def retrieve(self, query_vector, top_k):
                _, indices = self.index.search(np.array([query_vector]), top_k)
                keys = [self.docstore.inv[ind] for ind in indices[0]]
                vectors = [self.weaviate_store.get_vector(key) for key in keys]
                return keys, vectors
        return Retriever()

# Initialize the vectorstore
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), weaviate_vectorstore, {})

# Initialize the agent
from langchain.experimental import AutoGPT
from langchain.chat_models import ChatOpenAI

agent = AutoGPT.from_llm_and_tools(
    ai_name="🦜🔗CE_ARA",
    ai_role="Autonomous Research Assistant",
    tools=tools,
    llm=ChatOpenAI(temperature=0.5),
    memory=vectorstore.as_retriever()
)

# Set verbose to be true
agent.chain.verbose = True

# Define a function to gather user input for the goal and plans
def gather_input():
    # Get the goal for the research assistant
    goal = input("What is the goal for the Research Assistant? ")

    # Get the plans to achieve the goal
    plans = []
    for i in range(1, 6):
        plan = input(f"Enter plan {i} to achieve the goal (leave blank to stop entering plans): ")
        if plan == "":
            break
        plans.append(plan)

    return goal, plans

goal, plans = gather_input()

# Define a function to run the agent
def run_agent(agent, goal, plans, auto_task_count):
    # Initialize the auto task counter
    auto_task_counter = 0

    # Agent Run
    chain_order = []
    for plan in plans:
        chain_order.append(plan)
        input_text = f"{goal}+{','.join(chain_order)}"
        agent_response = agent.run([(input_text,)])[0]
        print("Research Assistant:", agent_response)

        auto_task_counter += 1
        if auto_task_counter > auto_task_count:
            approval = input("Should the agent continue with the next task? (y/n): ")
            if approval.lower() != 'y':
                print("The agent won't continue until you approve.")
                break

        # Check if the agent should pause for user response
        if auto_task_counter < auto_task_count:
            while True:
                user_input = input("Your response: ")
                if user_input.lower() == 'quit':  # Add condition to break the loop
                    return
                if user_input:
                    agent_response = pause_and_get_agent_response(agent, chain_order, user_input)
                    print("You:", user_input)
                    print("Research Assistant:", agent_response)

                    # Check if the user wants to use a tool
                    tool_response = check_and_use_tool(agent, tools, user_input)
                    if tool_response:
                        print("Research Assistant (Tool):", tool_response)

                else:
                    print("Waiting for your response...")

        if user_input.lower() == 'quit':  # Add condition to break the outer loop
            return

        # Check if the agent should finish after the current task
        if auto_task_counter >= auto_task_count:
            finish_response = input("Enter 'finish' to signal that you have finished all your objectives: ")
            if finish_response.lower() == 'finish':
                print("You have finished all your objectives.")
                return


def pause_and_get_agent_response(agent, chain_order, user_input):
    # Pause the LLM chain until user confirms to continue
    agent.llm.pause()
    # Generate the agent's response using the specified chain order
    chain_input = ','.join(chain_order + [user_input])
    agent_response = agent.run([(f"{goal}+{chain_input}",)])[0]
    agent.llm.resume()
    chain_order.append(user_input)
    return agent_response


def check_and_use_tool(agent, tools, user_input):
    # Check if the user wants to use a tool
    if user_input.lower() in [tool.name.lower() for tool in tools]:
        tool_name = user_input.lower()
        tool_input = input("Enter the input for the tool: ")
        # Use the selected tool
        tool_response = agent.run([(f"{tool_name}+{tool_input}",)])[0]
        return tool_response
    return None


# Run the agent
auto_task_count = int(input("1"))
run_agent(agent, goal, plans, auto_task_count)

