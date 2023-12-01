from langchain.agents import create_sql_agent
from langchain.agents import Tool, AgentType, initialize_agent, AgentExecutor
from functions.embeddings_demo import store
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI, SerpAPIWrapper, LLMChain,SQLDatabase, SQLDatabaseChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import (
    LLMSingleActionAgent,
    AgentOutputParser,
    ZeroShotAgent,
)
from langchain.chains import RetrievalQA
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
import os

#Otra forma desde una URL
from langchain.chat_models import ChatOpenAI


from dotenv import load_dotenv
load_dotenv()


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
print(os.environ["OPENAI_API_KEY"])

####FUENTES DE INFORMACION"#####################

# tiene que buscar en google y devolver el primer resultado todas las busquedas sera en Peru y en español
search = SerpAPIWrapper(serpapi_api_key=os.environ["SERPAPI_API_KEY"], params={
                        "engine": "google", "google_domain": "google.com", "gl": "pe", "hl": "es-419"})

#-------------------------------------------------
store_db = store()
retriever_pg = store_db.as_retriever(search_kwargs={"k": 1})

func_retriever = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    chain_type="stuff",
    retriever=retriever_pg,
    input_key="input",
)
##############################################


tools = [
    Tool(
        name="promociones",
        func=func_retriever.run,
        description="useful for when you need to answer about promociones of restaurants in miraflores",
    ),
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions",
    ),
    
]

prefix = """Have a conversation with a human, answering the following questions as best you can in spanish. You have access to the following tools:"""
suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)


memory = ConversationBufferMemory(memory_key="chat_history",input_key="input")

llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)


agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)

agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory
)

while True:
    question = str(input("Question: "))
    if question == "quit":
        break
    print(agent_chain.run(input=question))
