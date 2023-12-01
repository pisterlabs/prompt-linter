from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor


# Create a SQLDatabaseToolkit connection to the App_Patrol Database

db = SQLDatabase.from_uri("sqlite:////home/ec2-user/srtool/app_patrol.db")
toolkit = SQLDatabaseToolkit(db=db)

agent_executor = create_sql_agent(
    llm=OpenAI(temperature=0),
    toolkit=toolkit,
    verbose=True
)

#agent_executor.run("Describe nvd_cves table with a helpful summary abou the Severity column.")
#Take user input frrom the command line and run the agent on it
while True:
    gaurdrails = ("Do not use sql LIMIT in the results. ")
    user_input = gaurdrails + input("Enter a question or type 'exit' to quit: ")
    if user_input.lower() == 'exit':
        break

    agent_executor.run(user_input)


