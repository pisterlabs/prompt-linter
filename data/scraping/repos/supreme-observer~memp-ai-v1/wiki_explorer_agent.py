import os
from langchain import OpenAI, Wikipedia
from langchain.agents.react.base import DocstoreExplorer, Tool
from langchain.callbacks import get_openai_callback
from langchain.agents import initialize_agent



class WikiExplorerAgent:
    def __init__(self) -> None:
        docstore=DocstoreExplorer(Wikipedia())
        tools = [
            Tool(
                name="Search",
                func=docstore.search,
                description='search wikipedia'
            ),
            Tool(
                name="Lookup",
                func=docstore.lookup,
                description='lookup a term in wikipedia'
            )
        ]
    
        OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
        llm = OpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature=0
        )
        self.docstore_agent = initialize_agent(
                            tools,
                            llm,
                            agent="react-docstore",
                            verbose=True,
                            max_iterations=3
                        )
    def count_tokens(agent, query):
        with get_openai_callback() as cb:
            result = agent(query)
            print(f'Spent a total of {cb.total_tokens} tokens')
        return result
    def lookup_from_wikipedia(self,query):
        return self.count_tokens(self.docstore_agent, query)