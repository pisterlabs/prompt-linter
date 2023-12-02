import textwrap

import duckdb
from langchain import LLMMathChain
from langchain.agents import Tool, initialize_agent
from langchain.llms import OpenAIChat

from app.utils.data_query_chain import get_duckdb_data_query_chain
from app.utils.duckdb_query import run_sql, describe_table_or_view


def create_duckdb_llm_agent(duckdb_connection: duckdb.DuckDBPyConnection):
    """
    Create an agent that can answer questions about a duckdb database.
    Agents use an LLM to determine which actions to take and in what order.
    An action can either be using a tool and observing its output, or returning to the user.
    """
    # First, load the language model we're going to use to control the agent.
    llm = OpenAIChat(model_name="gpt-3.5-turbo", temperature=0)

    # Next, load tools that the agent can use to answer questions.
    llm_math_chain = LLMMathChain(llm=llm, verbose=True)
    db_op_chain = get_duckdb_data_query_chain(
        llm=llm, duckdb_connection=duckdb_connection
    )
    tools = [
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="Useful for when you need to answer questions about math",
        ),
        Tool(
            name="Show Tables",
            func=lambda _: run_sql(duckdb_connection, "show tables;"),
            description="Useful to show the available tables and views. Empty input required.",
        ),
        Tool(
            name="Describe Table",
            func=lambda table: describe_table_or_view(duckdb_connection, table),
            description="Useful to show the column names and types of a table or view. Use the table name as the input.",  # noqa: E501
        ),
        Tool(
            name="Data Op",
            func=lambda input: db_op_chain(
                {
                    "table_names": lambda _: run_sql(duckdb_connection, "show tables;"),
                    "input": input,
                }
            ),
            description=textwrap.dedent(
                """useful for when you need to operate on data and answer questions
            requiring data. Input should be in the form of a natural language question containing full context
            including what tables and columns are relevant to the question. Use only after data is present and loaded.
            """,  # noqa: E501
            ),
        ),
    ]

    # Finally, initialize the agent with the
    # tools, the language model, and the type of agent we want to use.
    agent = initialize_agent(
        tools,
        llm,
        # This agent uses the ReAct framework to determine which tool to use based solely on the
        # tool’s description.
        agent="zero-shot-react-description",
        agent_kwargs={
            "input_variables": ["input", "agent_scratchpad", "table_names"],
            "prefix": prompt_prefix,
            "suffix": prompt_suffix,
        },
        # return_intermediate_steps=True,
        verbose=True,
    )
    return agent


prompt_prefix = """Answer the following question as best you can by querying for data to back up
your answer. Even if you know the answer, you MUST show you can get the answer from the database.

Refuse to delete any data, or drop tables. When answering, you MUST query the database for any data.
Check the available tables exist first. Prefer to take single independent actions. Prefer to create views
of data as one action, then select data from the view.

Share the SQL queries you use to get the answer.

It is important that you use the exact phrase "Final Answer: " in your final answer.
List all SQL queries returned by Data Op in your final answer.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I don't know" as the answer.

You have access to the following data tables:
{table_names}

Only use the below tools. You have access to the following tools:
"""

prompt_suffix = """
It is important that you use the exact phrase "Final Answer: <Summary>" in your final answer.

Begin!

Question: {input}
Thought: I should look at the tables in the database to see what I can query.
{agent_scratchpad}"""
