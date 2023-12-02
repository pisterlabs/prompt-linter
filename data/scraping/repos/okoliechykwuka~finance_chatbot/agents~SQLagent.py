import os
from langchain.agents import create_csv_agent, AgentType
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.agents import create_sql_agent 
from langchain.agents.agent_toolkits import SQLDatabaseToolkit 
from langchain.sql_database import SQLDatabase 


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
    
    os.environ["DB_USER"] = username
    os.environ["DB_PASSWORD"] = password
    os.environ["DB_HOST"] = host
    os.environ["DB_NAME"] = database
    
    # db = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"
    db  = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"

    return db

def mysql_uri(username, password, host, port, database):
    try:
        assert username != None
        assert password != None
        assert host != None
        assert port != None
        assert database != None
    except:
        raise ValueError("Check all credential")
    port = int(port)
    # Set environment variables with user-entered values
    
    os.environ["DB_USER"] = username
    os.environ["DB_PASSWORD"] = password
    os.environ["DB_HOST"] = host
    os.environ["DB_NAME"] = database
    
    db = f"mysql+pymysql://{username}:{password}@{host}/{database}"
    
    return db


def build_sql_agent(llm,rdbs, **kwargs):
    #llm = OpenAI(temperature=0,model="text-davinci-003", streaming=True)
    print(rdbs.lower())
    print('----------------------------------------------------------------')
    if rdbs.lower() == 'postgres':
        uri = postgres_uri(**kwargs)
    elif rdbs.lower() == 'mysql':
        print(rdbs.lower())
        uri = mysql_uri(**kwargs)
    else:
        print('database not connected yet')
    
    db = SQLDatabase.from_uri(uri)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    sql_agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        handle_parsing_errors=True,
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