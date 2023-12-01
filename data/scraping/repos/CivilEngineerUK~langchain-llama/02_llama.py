import os
from dotenv import load_dotenv
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain


load_dotenv()

dburi = 'sqlite:///data/movies.db'
db = SQLDatabase.from_uri(dburi)

# load llm
# temperature = 0 is deterministic so good for factual data retrival
llm = OpenAI(temperature=0)

# build qa chain
db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)

db_chain.run("How many rows are in the movies table?")
db_chain.run("Describe the movies table.")
db_chain.run("What is the average rating of movies?")
db_chain.run("What is the average rating of movies with the genre 'Action'?")
db_chain.run("What is the average rating of movies with the genre 'Action' and 'Adventure'?")
