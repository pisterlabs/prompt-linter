import openai
import pandas as pd
import numpy as np
import pyreadr
import concurrent.futures
import os
import datetime


with open("secret", "r") as f: 
  openai.api_key = f.read()
output_dir = "all_codes"
  
development = pyreadr.read_r("data/development.rds")
development = development[None]
development = development
essays = development["response"].tolist()
ids = development["id"].tolist()
raters = np.arange(1, 6)

def code_essay(id, essay):
  response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
              {"role": "system", "content": "You are an admissions officer that reads students essays about personal qualities and gives them a score between 0 and 100 for seven personal qualities. Zero means there is no evidence for the personal quality in the text, 100 means that you are very sure that the students shows that quality. Answer with a score and an explanation for each. The qualities are: Prosocial Purpose: Helping others, wanting to help others, consideration of the benefits to others, mention of reasons for helping others, or reflection on how enjoyable or rewarding it is to help others. Leadership: Serving in a leadership role, commenting on what he or she did in his or her capacity as a leader, or discuss the value, meaning, or importance of leadership. Learning: Improving, learning, or developing knowledge, skills, or abilities. Goal pursuit: Having a goal and/or a plan. Intrinsic motivation: Describing the activity as enjoyable or inter- esting. Liking the activity or identifying with it. Teamwork: Working with or learning from others. Valuing what fellow participants bring to the activity. Perseverance: Persisting in the face of challenge especially for extended periods of time. Answer with the personal quality, folowed by a vertical line, the score, a vertical line, and an explanation. Put each personal quality on a new line."},
              {"role": "user", "content": essay},
          ]
  )
  result = response.choices[0].message.content
  now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
  with open(f"{output_dir}/{id}_{now}.txt", "w") as f:
    f.write(result)


with concurrent.futures.ThreadPoolExecutor() as executor:
  executor.map(code_essay, ids, essays)

data = pd.DataFrame()
for file in os.listdir(output_dir):
  # i = pd.read_csv(f"raw_results/{file}", sep="\\||:", header=None).assign(id=file.replace(".0.txt", ""))
  i = pd.read_csv(f"{output_dir}/{file}", sep="xxxxx", header=None).assign(id=file.replace(".0|.txt", ""))
  data = pd.concat([data, i])

data

data.columns = ["raw", "id"]
data

data["id"] = data["id"].str.replace(".0_", "_")
data["id"] = data["id"].str.replace(".txt", "")

# split id col into id and raters
data[["id", "rater"]] = data["id"].str.split("_", expand=True)
data.id.tolist().unique
data.to_csv("data/gpt_ratings_3.csv", index=False)

