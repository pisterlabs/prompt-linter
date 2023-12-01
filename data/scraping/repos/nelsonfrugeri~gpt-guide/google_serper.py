import os

from langchain.utilities import GoogleSerperAPIWrapper
from langchain.llms.openai import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

def main():
    self_ask_with_search = initialize_agent(
        [
            Tool(
                name="Intermediate Answer",
                func=GoogleSerperAPIWrapper().run,
                description="useful for when you need to ask with search",
            )
        ], OpenAI(temperature=os.getenv("OPENAI_PARAM_TEMPERATURE")),
            agent=AgentType.SELF_ASK_WITH_SEARCH,
            verbose=os.getenv("GOOGLE_SERPER_API_PARAM_VERBOSE")
    )

    print(self_ask_with_search.run(input("Hello, what can I do for you?: ")))

if __name__ == "__main__":
    main()