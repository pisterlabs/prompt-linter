import langchain
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent, initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.schema import SystemMessage

from ecommerce_agent.callbacks.chat_model_callback_handler import ChatModelCallbackHandler
from ecommerce_agent.tools.describe_tables_tool import describe_tables_tool
from ecommerce_agent.tools.sql_query_runner_tool import sql_query_tool
from ecommerce_agent.tools.write_html_report_tool import write_html_report_tool
from ecommerce_agent.utils.list_tables_sql_runner import list_tables

# Load environment variables
load_dotenv()

langchain.debug = True

# Instantiate chat model callback handler
chat_model_callback_handler = ChatModelCallbackHandler()

# Instantiate chat model
open_ai_chat_llm = ChatOpenAI(callbacks=[chat_model_callback_handler])

# Get list of tables in the database
tables_in_db = list_tables()
print(tables_in_db)

# Instantiate prompt template
chat_prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(
            content=(
                "You are an AI assistant that has access to a SQLite database.\n"
                f"The database has tables of: {tables_in_db}\n"
                "Do not make any assumptions about what tables exist or "
                "what columns exist. Instead use the 'describe_tables' function."
            )),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

# Instantiate memory to pass to AgentExecutor
# This way agent executor has context preserved between subsequent agent executor runs
agent_executor_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define tools list
tools_list = [sql_query_tool, describe_tables_tool, write_html_report_tool]

# Instantiate agent
open_ai_functions_agent = OpenAIFunctionsAgent(
    llm=open_ai_chat_llm,
    prompt=chat_prompt,
    tools=tools_list)

# Instantiate agent executor
agent_executor = AgentExecutor(
    agent=open_ai_functions_agent,
    verbose=True,
    tools=tools_list,
    memory=agent_executor_memory)

# Alternative way to instantiate agent executor
# agent_executor = initialize_agent(llm=open_ai_chat_llm,
#                                   tools=tools_list,
#                                   agent=AgentType.OPENAI_FUNCTIONS,
#                                   verbose=True)

# Run agent executor
agent_executor(
    "Summarize the top 5 most popular products. Write the results to a HTML report file in the 'reports' directory.")

# Second run
agent_executor("Repeat the same process for carts")
