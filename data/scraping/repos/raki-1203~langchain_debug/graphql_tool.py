import os

from langchain import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.utilities import GraphQLAPIWrapper

import config as c

os.environ['OPENAI_API_KEY'] = c.OPENAI_API_KEY
os.environ['SERPAPI_API_KEY'] = c.SERPAPI_API_KEY
os.environ['SERPER_API_KEY'] = c.SERPER_API_KEY
os.environ['GOOGLE_API_KEY'] = c.GOOGLE_API_KEY
os.environ['GOOGLE_CSE_ID'] = c.GOOGLE_CSE_ID


if __name__ == '__main__':
    llm = OpenAI(temperature=0)

    # BaseGraphQLTool instance with the Star Wars API endpoint
    tools = load_tools(tool_names=['graphql'],
                       graphql_endpoint="https://swapi-graphql.netlify.app/.netlify/functions/index",
                       llm=llm)

    agent = initialize_agent(tools=tools,
                             llm=llm,
                             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                             verbose=True)

    graphql_fields = """allFilms {
    films {
        title
        director
        releaseDate
        speciesConnection {
            species {
                name
                classification
                homeworld {
                    name
                }
            }
        }
    }
}
"""

    suffix = "Search for the titles of all the stawars films stored in the graphql database that has this schema "

    agent.run(suffix + graphql_fields)





