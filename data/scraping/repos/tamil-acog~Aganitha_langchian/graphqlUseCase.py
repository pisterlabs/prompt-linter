
from langchain import OpenAI
from langchain.chains.graphQL import prompt
from langchain.chains.graphQL.base import GraphQLChain
from langchain.graphQL import GraphQL


db: GraphQL = GraphQL(graphql_url="https://api.platform.opentargets.org/api/v4/graphql")

llm: OpenAI = OpenAI(model_name="gpt-3.5-turbo", temperature=0)

db_chain: GraphQLChain = GraphQLChain(llm=llm, database=db, verbose=True, prompt=prompt.prompt)
query = str(input("Hi, You can now talk to open targets database. \n"))
db_chain.run(query)


