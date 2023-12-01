from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
from langchain.prompts.prompt import PromptTemplate

_DEFAULT_TEMPLATE = """Given an input question, first create a
syntactically correct {dialect} query to run, then look at the results of
the query and return the answer.

Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"
Seaborn or matplotlib complete code to plot the results on a chart or table: "Seaborn code here"

Only use the following tables:

events, artists, pages
the artits per event are in the events table
Question: {input}"""

PROMPT = PromptTemplate(
    input_variables=["input", "dialect"], template=_DEFAULT_TEMPLATE
)

db = SQLDatabase.from_uri("sqlite:///docs/data_clean.db")
llm = OpenAI(temperature=0)

db_chain = SQLDatabaseChain(database=db, llm=llm, verbose=True, prompt=PROMPT)
while True:
    query = input("Query: ")
    if not query:
        break
    print(db_chain.run(query))
