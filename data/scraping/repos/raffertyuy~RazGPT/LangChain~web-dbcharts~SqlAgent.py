from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent

class SqlAgent:
    """
    A utility class for querying an SQL database using an agent and returning the response as a string.

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

        prompt = ("""For the following query, if it requires drawing a table, reply as follows:
{"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

If the query requires creating a bar chart, reply as follows:
{"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

If the query requires creating a line chart, reply as follows:
{"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

There can only be two types of chart, "bar" and "line".

If it is just asking a question that requires neither, reply as follows:
{"answer": "answer"}
Example:
{"answer": "The title with the highest rating is 'Gilead'"}

If you do not know the answer, reply as follows:
{"answer": "I do not know."}

Return all output as a string.

All strings in "columns" list and data list, should be in double quotes,

For example: {"columns": ["title", "ratings_count"], "data": [["Gilead", 361], ["Spider's Web", 5164]]}

Lets think step by step.

Below is the query.

Query: """ + query)

        # Run the prompt through the agent.
        response = self.agent.run(prompt)

        # Convert the response to a string.
        return response.__str__()
        