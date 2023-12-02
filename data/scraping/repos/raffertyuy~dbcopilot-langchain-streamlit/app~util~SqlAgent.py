from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent

from . import Prompts

class SqlAgent:
    """
    A LangChain SQL Agent helper class.

    Args:
        db: The name of the database to connect to.
        llm: The language model to use for the agent.
        verbose: Whether to print verbose output.

    Attributes:
        agent: The SQL agent used for querying the database.

    Methods:
        Query(query): Query the agent and return the response as a string.
    """

    def __init__(self, db, llm, verbose=False):
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        self.agent = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=verbose
        )

    # Functions
    def Query(self, query):
        """
        Query an agent and return the response as a string.

        Args:
            query: The query to ask the agent.

        Returns:
            The response from the agent as a string.
        """

        #prompt = Prompts.QUERY_PROMPT + query
        prompt = query
        response = self.agent.run(prompt)

        return response.__str__()