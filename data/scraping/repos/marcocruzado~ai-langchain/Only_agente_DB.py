from langchain import (
    LLMMathChain,
    OpenAI,
    SerpAPIWrapper,
    SQLDatabase,
    SQLDatabaseChain,
)
from langchain.schema import SystemMessage
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate, StringPromptTemplate
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
import os
from dotenv import load_dotenv
load_dotenv()


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
print(os.environ["OPENAI_API_KEY"])

#### FUENTES DE INFORMACION"#####################

# tiene que buscar en google y devolver el primer resultado todas las busquedas sera en Peru y en español
search = SerpAPIWrapper(serpapi_api_key=os.environ["SERPAPI_API_KEY"], params={
                        "engine": "google", "google_domain": "google.com", "gl": "pe", "hl": "es-419"})

# -------------------------------------------------
# postgresql+psycopg2://pguser:password@localhost:5433/doc_search
db = SQLDatabase.from_uri(
    'postgresql+psycopg2://postgres:root@localhost:5432/postgres')

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613") # gpt-3.5-turbo-0613
############PROMPT TEMPLATE########################

template = """
CONTEXT :
You are an agent of the "EL COMERCIO subscribers club", who answers the questions of the users, you will provide all the relevant information and that is very consistent with this question {input} that the user is requesting,
Your responses should be addressed to people who are subscribers to "CLUB EL COMERCIO", such as:

- You currently have these discounts at the establishment and you can call this number to request delivery

- The validity of the discount and/or promotion is valid only until

- The opening hours that apply to said discount are from ... to ....

and always keeping these possible answers as a reference, they will not necessarily be the same, but at least what resembles a conversation with a digital agent.

using the tools you have access to:

You will carry out the corresponding analysis and generate the best answer by following these steps.

Steps:
1.- You will look for the answers to the question within the database.
2.- then you will analyze the answer provided.
3.- And if it is not related or you did not find anything then, you will search once more until you have a satisfactory answer.
4.- You will provide the answer to the user, friendly and very educated and always in Spanish

If after carrying out the aforementioned procedure we do not have a satisfactory answer, then in a very elegant way, tell the subscriber that, for the moment, you do not have the information consulted and you will take it into account for future implementations and always in Spanish.

Your answer:"""

#############AGENTE###############################

db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

""" tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions",
    ),
    Tool(
        name="FooBar-DB",
        func=db_chain.run,
        description="useful for when you need to answer questions about FooBar. Input should be in the form of a question containing full context",
    ),
] """

tools = [
    Tool(
        name="suscriptores",
        func=db_chain.run,
        description="Useful for when you need to answer questions about discounts or promotions of all types of establishment and if you do not have the information then kindly answer that, we will be incorporating it soon.",
    ),
]

agent_kwargs = {
    "system_message": SystemMessage(content="You are an expert SQL data analyst."),
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
}
memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)


while True:
    print("Enter a message:")
    message = input()
    print(">"*12,agent.run(input=message),"<"*12)