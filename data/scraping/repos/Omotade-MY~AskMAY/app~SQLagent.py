import os
from langchain.agents import create_csv_agent, AgentType
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.agents import create_sql_agent 
from langchain.agents.agent_toolkits import SQLDatabaseToolkit 
from langchain.sql_database import SQLDatabase 
os.environ["LANGCHAIN_TRACING"] = "true"


#os.environ['OPENAI_API_KEY'] = "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"

def postgres_uri(username, password, host, port, database):
    try:
        assert username != None
        assert password != None
        assert host != None
        assert port != None
        assert database != None
    except:
        raise ValueError("Check all credential")
    port = int(port)
    
    pg_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"

    return pg_uri


def build_sql_agent(llm,rdbs, **kwargs):
    #llm = OpenAI(temperature=0,model="text-davinci-003", streaming=True)
    if rdbs.lower() == 'postgres':
        uri = postgres_uri(**kwargs)
    
    db = SQLDatabase.from_uri(uri)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    sql_agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    return sql_agent

def sql_as_tool(agent):
    return Tool.from_function(
            name = "sql_retrieval_tool",
            func=agent.run,
            description= "Use this tool if you need to run queries against the database.",
        )
#sql_agent = build_sql_agent()
#message = "what is the `total score` for 'Sunday Nwoye' added to 'Helen Opayemi'"
#sql_agent.run(input=message)


"""if chroma:
            context = [c.page_content for c in chroma.similarity_search(
                user_input, k=10)]
            user_input_w_context = PromptTemplate(
                template=PROMPT_TEMPLATE,
                input_variables=["context", "question"]) \
                .format(
                    context=context, question=user_input)
            """