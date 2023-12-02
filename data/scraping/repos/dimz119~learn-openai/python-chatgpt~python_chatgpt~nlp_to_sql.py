import openai
import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import text

openai.api_key = os.getenv("OPENAI_API_KEY")

df = pd.read_csv("sample/temp_with_date.csv")
# print(df)
# print(df.groupby("location").min("temperature"))

db_con = create_engine('sqlite:///:memory:', echo=True)
data = df.to_sql(name='temperature', con=db_con)
print(data)

with db_con.connect() as conn:
    result = conn.execute(text("SELECT location, MIN(temperature) as min_temp FROM temperature GROUP BY location"))

print(result.all())