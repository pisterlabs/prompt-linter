from langchain import OpenAI
from langchain.postgres import PostgreSQL
from langchain.chains.postgre_sql.base import PostgreSQLChain
from langchain.agents import initialize_agent, Tool
from langchain.utilities import SearxSearchWrapper


search: SearxSearchWrapper = SearxSearchWrapper(searx_host="https://agansearch.hub-dev.aganitha.ai/",
                                                headers={"authorization": "Basic YWNvZzpnMGJiIWVkeWcwMGs="})


llm: OpenAI = OpenAI(model_name="text-davinci-003", temperature=0)


db: PostgreSQL = PostgreSQL(user="postgres", password="guNagaNa1", host="127.0.0.1", port="5432", database="aact",
                            schema="ctgov",
                            include_tables=["studies", "interventions", "sponsors"])

db_chain: PostgreSQLChain = PostgreSQLChain(llm=llm, database=db, verbose=True)
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about "
                    "current events. You should ask targeted questions"
    ),
    Tool(
        name="aact db",
        func=db_chain.run,
        description="useful for when you need to answer questions about clinical trials"
    )
]

query_agent = initialize_agent(tools, llm, agent="zero-shot-react-description")

# query_agent.run("List the sponsors of Haemophilia clinical trials in Phase 3")
query_agent.run("Give the overall status of NCT03947632?")
# query_agent.run("Explain me about the aact database.")
