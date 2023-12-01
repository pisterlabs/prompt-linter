import sqlite3
import pandas as pd
import os

df = pd.read_csv('data.csv')

conn = sqlite3.connect('auxel_db.sqlite3')

# df.to_sql('sales_data', conn, if_exists='replace')

# LangChain SQL Agent

from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
import sqlite3

os.environ['OPENAI_API_KEY'] = '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'

input_db = SQLDatabase.from_uri('sqlite:///auxel_db.sqlite3')

llm_1 = OpenAI(temperature=0)

db_agent = SQLDatabaseChain(llm=llm_1,
                            database=input_db,
                            verbose=True)

out = db_agent.run('create new table for each of the category')
print(type(out))

