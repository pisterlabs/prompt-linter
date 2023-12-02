import openai
import pandas as pd
import numpy as np
import concurrent.futures
import os
import datetime


with open("secret", "r") as f: 
  openai.api_key = f.read()
source_data = "data/wang/wang.csv"
text_var = "text"
output_dir = "data/wang/all_codes"

# create output dir if it doesn't exist
if not os.path.exists(output_dir):
  os.makedirs(output_dir)

TEMP = 1
prompt = (
  """
  You are a trained research assistant that CATEGORIZES organizations self-reported sustainability initiatives into one of 16 categories. 
  The categories are:
  Respond with the name of the category: for example: 'Creating sustainable products and processes'. 
  Here are the categories followed by a short description or example.
  Creating sustainable products and processes: Created products that used fewer nonrenewable resources (e.g., cosmetics without petroleum-based ingredients)
  Embracing innovation for sustainability: Used reflective roofing to reduce cooling costs
  Changing how work is done: Optimized processes and equipment (e.g., shipping routes, containers) to reduce fuel usage
  Choosing responsible alternatives: Switched to renewable energy sources for manufacturing facilities
  Monitoring environmental impact: Measured the amount of material waste produced
  Preventing pollution: Installed scrubbers to reduce greenhouse gas emissions
  Strengthening ecosystems: Planted a tree for each customer that signed up for e-billing
  Reducing use: Reduced electricity use (e.g., installing motion sensors, using energy-efficient lighting, shutting down equipment not in use)
  Reusing: Reused supplies several times (e.g., shipping crates, packaging materials)
  Recycling: Distributed recycling bins throughout several locations
  Repurposing: Used rainwater to supply water to several facilities
  Encouraging and supporting others: Provided incentives to encourage employees to use alternate modes of commuting (e.g., biking, public transport)
  Educating and training for sustainability: Trained employees how to perform their work in an environmentally responsible manner
  Instituting programs and policies: Founded an industry alliance to establish a uniform set of environmental standards
  Putting environmental interests first: Reduced temperatures in office buildings to save energy
  Lobbying and activism: Participated in the development of local and federal environmental standards and regulations  
  Sort the text I am going to tell you into only one of these 16 categories.
  Respond only with the name of the category, and do not include any extra information.
  """)
  

data = pd.read_csv(source_data)
data[text_var] = data.description_behavior + " " + data.consequences 
data["id"] = np.arange(1, len(data)+1)

essays = data[['id','text']].drop_duplicates().text.tolist()
ids = data[['id','text']].drop_duplicates().id.tolist()
raters = np.arange(1, 6)

def code_essay(id, essay):
  response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      temperature=TEMP,
      messages=[
              {"role": "system", "content": prompt},
              {"role": "user", "content": essay},
          ]
  )
  result = response.choices[0].message.content
  now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
  with open(f"{output_dir}/{id}_{now}_temp{TEMP}.txt", "w") as f:
    f.write(result)



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


data.to_csv("data/wang/gpt_ratings.csv", index=False)

