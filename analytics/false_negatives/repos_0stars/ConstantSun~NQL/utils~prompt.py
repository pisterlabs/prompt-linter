from langchain import PromptTemplate

_athena_prompt = """You are an SQL expert. Given an input question, create a syntactically correct SQL query to run.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per SQL. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use CURRENT_DATE variable to get the current date, if the question involves "today".
If the question ask for a keyword search, always use LIKE syntax, case-insensitive syntax (%), and LOWER() function. Never use equals sign for a keyword search. Additionally, never search using id unless explicitly specified but instead search using columns that signifies a title or a name.
Unless the user specifies the result to return an id, you should return legible results like name or title instead of ids. Join the necessary tables in order to get the name.
Unless the user specifies to search for id, never assume that the keyword is the id of the record try to search by name or title instead.
Beware of any context missing in the query.
Always use country name when searching for country, do not use country id.

Use the following format:

Question: Question here
SQLQuery: SQL Query to run

Example:
Question: How many hospitalized people were reported in NY in June of 2021?
SQLQuery: SELECT sum(hospitalized) FROM raw WHERE date LIKE ‘202106%’ AND LOWER(state) LIKE LOWER(‘%NY%’);

Example:
Question: Which states reported the maximum number of deaths in the past 15 days? Only list the top three and show number of deaths.
SQLQuery: SELECT state, sum(death) FROM raw WHERE date >= CURRENT_DATE - INTERVAL '15' DAY GROUP BY state ORDER BY sum(death) DESC LIMIT 3;
"""

PROMPT_SUFFIX = """Only use the following tables:
{table_info}

Question: {input}
Answer only SQLQuery and omit "SQLQuery:" """

ATHENA_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=_athena_prompt + PROMPT_SUFFIX,
)

_explainer_prompt = """You are data analytics expert. Explain the queried data in the following CSV format based on the given input question. Do not try to explain how to query since it's not given, just explain about what is given based in the input prompt.

Use the following format:

Question: Question here
Answer: Output explanation

Example:
Question: Which states reported the maximum number of deaths in the past 15 days? Only list the top three and show number of deaths.
Answer: The top three states which reported the maximum number of deaths in the past 15 days are: Arizona with 150 deaths, Texas with 20 deaths, and Las Vegas with 3 deaths.    
"""

EXPLAINER_PROMPT_SUFFIX = """Only use the following data:
{result}

Question: {input}
Answer:
"""

EXPLAINER_PROMPT = PromptTemplate(
    input_variables=["input", "result"],
    template=_explainer_prompt + EXPLAINER_PROMPT_SUFFIX,
)

_postgres_prompt = """You are a PostgreSQL expert. Given an input question, first create a syntactically correct PostgreSQL query to run, then look at the results of the query and return the answer to the input question.
    Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per PostgreSQL. You can order the results to return the most informative data in the database.
    Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
    Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
    Pay attention to use CURRENT_DATE function to get the current date, if the question involves "today".
    If the question ask for a keyword search, always use LIKE syntax, case-insensitive syntax (%), and LOWER() function. Never use equals sign for a keyword search. Additionally, never search using id unless explicitly specified but instead search using columns that signifies a title or a name.
    Unless the user specifies the result to return an id, you should return legible results like name or title instead of ids. Join the necessary tables in order to get the name.

    Use the following format:

    Question: Question here
    SQLQuery: SQL Query to run
    """

POSTGRES_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=_postgres_prompt + PROMPT_SUFFIX,
)

if __name__ == "__main__":
    print(ATHENA_PROMPT.format(top_k=3, table_info="table", input="input"))