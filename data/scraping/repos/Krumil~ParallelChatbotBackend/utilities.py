from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain.retrievers.tavily_search_api import TavilySearchAPIRetriever

import os

load_dotenv()

os.getenv("LANGCHAIN_TRACING_V2")
os.getenv("LANGCHAIN_ENDPOINT")
os.getenv("LANGCHAIN_API_KEY") 
os.getenv("LANGCHAIN_PROJECT")  

os.getenv("TAVILY_API_KEY")


DEPLOYMENT_ENV = os.environ.get('DEPLOYMENT_ENV', 'DEVELOPMENT')

if DEPLOYMENT_ENV == 'PRODUCTION':
	base_directory = "/var/data/embeddings/"
else:
	base_directory = ".\\embeddings\\"

print(os.getcwd())


def check_read_access(path):
    if os.access(path, os.R_OK):
        print(f"The script has read access to {path}")
    else:
        print(f"The script does NOT have read access to {path}")


openai_api_key = os.environ["OPENAI_API_KEY"]

tools = []

def initialize_tools():
	global tools
	pdf_vectorstore = Chroma(persist_directory=os.path.join(base_directory, "pdf_chroma_db"), embedding_function=OpenAIEmbeddings())
	pdf_retriever = pdf_vectorstore.as_retriever()

	csv_vectorstore = Chroma(persist_directory=os.path.join(base_directory, "csv_chroma_db"), embedding_function=OpenAIEmbeddings())
	csv_retriever = csv_vectorstore.as_retriever()

	gitbook_vectorstore = Chroma(persist_directory=os.path.join(base_directory, "gitbook_chroma_db"), embedding_function=OpenAIEmbeddings())
	gitbook_retriever = gitbook_vectorstore.as_retriever()

	web_retriever = TavilySearchAPIRetriever(k=4)

	main_tool = create_retriever_tool(
		pdf_retriever, 
		"parallel_tcg",
		"Documents detail 'Parallel', a post-apocalyptic trading card game with five human factions, and its 'Echo Replication' feature allowing creation of Echo cards using in-game resources."
	)
	csv_tool = create_retriever_tool(
		csv_retriever,
		"cards_database",
		"Useful for answering questions about cards"
	)
	gitbook_tool = create_retriever_tool(
		gitbook_retriever,
		"echelon_docs",
		"Useful for answering questions about PRIME, Echelon, and the anything related to the economics of the Parallel ecosystem"
	)
	web_tool = create_retriever_tool(
		web_retriever,
		"web",
		"Search the web about information related to Parallel TCG"
	)
	tools = [main_tool, csv_tool, gitbook_tool, web_tool]



def initialize_bot(llm):
	global tools
	# Memory Component
	memory_key = "history"
	memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm, max_history=2, max_token_limit= 3000)

	# Prompt Template
	system_message = SystemMessage(
		content=(
			"Do your best to answer the questions about Parallel TCG. "
			"Feel free to use any tools available to look up relevant information."
		)
	)
	prompt = OpenAIFunctionsAgent.create_prompt(
		system_message=system_message,
		extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)]
	)

	# Agent
	agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)

	# Agent Executor
	return AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, return_intermediate_steps=True)    
