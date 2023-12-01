## LENR_AGPT_v0_2.py
## Added the following: 
#1. dotenv importer 
#2. Read the PDF file and extract the text
#3. User/Agent prompt communications (work in progress)

import os
from dotenv import load_dotenv
import requests
from PyPDF2 import PdfFileReader
from io import BytesIO
import pickle
import time


# Load the environment variables from .env file
load_dotenv()

# Get the API keys from environment variables
serpapikey = os.environ.get('SERPAPI_API_KEY')
openaikey = os.environ.get('OPENAI_API_KEY')
wolframalphaapikey = os.environ.get('WOLFRAM_ALPHA_APPID')
googleapikey = os.environ.get('GOOGLE_API_KEY')
googlesearch = os.environ.get('GOOGLE_CSE_ID')

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
    WriteFileTool(),
    ReadFileTool(),
    Tool(
        name="read_and_store_local_pdf",
        func=read_and_store_pdf_tool.run,
        description="Useful for reading a PDF from local filesystem and storing its content."
        
    ),
       
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
  
]

# Define your embedding model
embeddings_model = OpenAIEmbeddings()

# Initialize the vectorstore as empty
embedding_size = 1536  # openai embeddings has 1536 dimensions
index = faiss.IndexFlatL2(embedding_size)  # Index that stores the full vectors and performs exhaustive search
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

# Initialize the agent
from langchain.experimental import AutoGPT
from langchain.chat_models import ChatOpenAI

agent = AutoGPT.from_llm_and_tools(
    ai_name="🦜🔗ARA",
    ai_role=" Autonomous Research Assistant",
    tools=tools,
    llm=ChatOpenAI(temperature=0.7),
    memory=vectorstore.as_retriever()
)

# Set verbose to be true
agent.chain.verbose = True

# Agent User Communication
print("You are now communicating with the Research Assistant.")
print("Enter 'quit' to stop the conversation.")

# Get the goal for the research assistant
goal = input("What is the goal for the Research Assistant? ")

# Get the plans to achieve the goal
plans = []
for i in range(1, 6):
    plan = input(f"Enter plan {i} to achieve the goal (leave blank to stop entering plans): ")
    if plan == "":
        break
    plans.append(plan)

# Display the goal and plans
print(f"Goal for the Research Assistant: {goal}")
print("Plans to achieve the goal:")
for i, plan in enumerate(plans):
    print(f"{i+1}. {plan}")

auto_task_count = int(input("How many tasks do you want the agent to perform automatically? "))
auto_task_counter = 1

# Agent Run
chain_order = []
for plan in plans:
    chain_order.append(plan)
    input_text = f"{goal}+{','.join(chain_order)}"
    agent_response = agent.run([(input_text,)])[0]
    print("Research Assistant:", agent_response)

    # Insert RLHF logic here, or call a function that handles RLHF
    # Example:
    # rlhf_response = perform_rlhf(agent, chain_order)
    # print("RLHF Response:", rlhf_response)

    # Pause and wait for user confirmation to continue
    continue_response = input("Press 'Enter' to continue to the next task or type 'quit' to stop: ")
    if continue_response.lower() == 'quit':
        break

    # Check if the agent should pause for user response
    if auto_task_counter < auto_task_count:
        while True:
            # Wait for user input and send it to the agent
            user_input = input("Your response: ")
            if user_input == 'quit':  # Add condition to break the loop
                break
            if user_input:
                # Pause the LLM chain until user confirms to continue
                agent.llm.pause()
                # Generate the agent's response using the specified chain order
                chain_input = ','.join(chain_order + [user_input])
                agent_response = agent.run([(f"{goal}+{chain_input}",)])[0]
                print("You:", user_input)
                print("Research Assistant:", agent_response)
                agent.llm.resume()
                chain_order.append(user_input)

                # Check if the user wants to use a tool
                if user_input.lower() in [tool.name.lower() for tool in tools]:
                    tool_name = user_input.lower()
                    tool_input = input("Enter the input for the tool: ")
                    # Use the selected tool
                    tool_response = agent.run([(f"{tool_name}+{tool_input}",)])[0]
                    print("Research Assistant (Tool):", tool_response)

                break
            else:
                print("Waiting for your response...")

    if user_input == 'quit':  # Add condition to break the outer loop
        break

    # Check if the agent should finish after the current task
    if auto_task_counter >= auto_task_count:
        finish_response = input("Enter 'finish' to signal that you have finished all your objectives: ")
        if finish_response.lower() == 'finish':
            print("You have finished all your objectives.")
            break
