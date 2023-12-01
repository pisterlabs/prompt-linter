import openai
import pandas as pd
import numpy as np
import concurrent.futures
import os
import datetime


with open("secret", "r") as f: 
  openai.api_key = f.read()
output_dir = "data/rohrer/all_codes"
TEMP = 1
  
data = pd.read_csv("data/rohrer/Rohrer Data Long.csv")

essays = data[['id','text']].drop_duplicates().text.tolist()
ids = data[['id','text']].drop_duplicates().id.tolist()
raters = np.arange(1, 6)

def code_essay(id, essay):
  response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      temperature=TEMP,
      messages=[
              {"role": "system", "content": "You are a trained reserach assistant that reads participant responses about strategies participants may use to improve their life. Participant responses are in german. Rate each response for three criteria: (1) idea, say 1 if the column contains an idea for improving one's life or 0 if not. (2) active, only relevant if idea == 1. Say 1 if  the text describe an active strategy to improve one's life (e.g., go out and make new friends), or 0 if it describes a passive event that should improve one's life (e.g., win the lottery). (3) social, only relevant if active == 1, say 1 if the text describes a social strategy (e.g. make new friends) or 0 if it is nonsocial (e.g., study harder). If unsure, respond with a decimal value between 0-1. Here's an example output. idea|1|active|0|social|0. Answer with the criteria, folowed by a vertical line, the score. "},
              {"role": "user", "content": essay},
          ]
  )
  result = response.choices[0].message.content
  now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
  with open(f"{output_dir}/{id}_{now}_temp{TEMP}.txt", "w") as f:
    f.write(result)

code_essay(ids[0], essays[0])

with concurrent.futures.ThreadPoolExecutor() as executor:
  executor.map(code_essay, ids, essays)

data = pd.DataFrame()
for file in os.listdir(output_dir):
  # i = pd.read_csv(f"raw_results/{file}", sep="\\||:", header=None).assign(id=file.replace(".0.txt", ""))
  i = pd.read_csv(f"{output_dir}/{file}", sep="xxxxx", header=None).assign(id=file)
  data = pd.concat([data, i])

data


data.columns = ["raw", "id"]
data

data["id"] = data["id"].str.replace(".txt", "")

# split id col into id and raters
data[["id", "rater", "temp"]] = data["id"].str.split("_", expand=True)

data
data.to_csv("data/rohrer/gpt_ratings.csv", index=False)

