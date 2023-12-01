from langchain.chains.graphQL.base import GraphQLChain
from langchain.graphQL import GraphQL
from langchain import OpenAI
from langchain.postgres import PostgreSQL
from langchain.chains.postgre_sql.base import PostgreSQLChain
from langchain.agents import initialize_agent, Tool
from langchain.utilities import SearxSearchWrapper
from langchain.chains.graphQL import prompt

search: SearxSearchWrapper = SearxSearchWrapper(searx_host="https://agansearch.hub-dev.aganitha.ai/",
                                                headers={"authorization": "Basic YWNvZzpnMGJiIWVkeWcwMGs="})

db: PostgreSQL = PostgreSQL(user="postgres", password="guNagaNa1", host="localhost", port="5432", database="aact",schema="ctgov",
                 include_tables=["studies", "interventions", "sponsors", "conditions"])

graphdb: GraphQL = GraphQL(graphql_url="https://api.platform.opentargets.org/api/v4/graphql")

llm: OpenAI = OpenAI(model_name="gpt-3.5-turbo", temperature=0)

graphdb_chain: GraphQLChain = GraphQLChain(llm=llm, database=graphdb, verbose=True, prompt=prompt.prompt)

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
    ),
    Tool(
        name="graphdb",
        func=graphdb_chain.run,
        description="useful for when you need to answer questions from open targets"
    )
]

query_agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
query = str(input("Hi, You can now talk to open targets database. \n"))
#graphdb_chain.run(query)
query_agent.run(query)
