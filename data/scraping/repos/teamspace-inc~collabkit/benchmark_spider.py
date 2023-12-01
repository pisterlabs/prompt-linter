import json
import sqlite3
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain

def process_db(db_name, query, index):
  con = sqlite3.connect("spider/database/{}/{}.sqlite".format(db_name, db_name))
  db = SQLDatabase.from_uri("sqlite:///spider/database/{}/{}.sqlite".format(db_name, db_name), sample_rows_in_table_info=2)

  def runQueryGPT4(query):
      llm = OpenAI(temperature=0, model_name="gpt-4")
      db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True, return_intermediate_steps=True, top_k=20)
      output = db_chain(query)
      print(output)
      return output
  
  cur = con.cursor()
  dataset = cur.execute(query['query']).fetchall().__str__()
  gpt = runQueryGPT4(query['question'])
  if(dataset == gpt['intermediate_steps'][1]):
    print("Correct")
    return 1
  else:
    print("Incorrect")
    with open("incorrect/{}.txt".format(db_name), "a") as f:
      f.write("Index: " + str(index) + "\n")
      f.write("Question: " + query['question'] + "\n")
      f.write("Spider-Query: " + query['query'] + "\n")
      f.write("GPT-Query: " + gpt['intermediate_steps'][0] + "\n")
      f.write("Expected-Data-Spider: " + dataset + "\n")
      f.write("GPT-Data: " + gpt['intermediate_steps'][1] + "\n")
      f.write("\n")
    return 0

with open("spider/train_spider.json", "r") as f:
  arr = json.load(f)

count = 0
correct = 0
incorrect = 0
checkpoint = 0
for obj in arr:
   count+=1
   if(count > checkpoint):
    ans = process_db(obj['db_id'], obj, count)
    if(ans == 1):
        correct+=1
    else:   
        incorrect+=1
    print("Count:",count, "\n")
    print("Correct:", correct, "\nIncorrect:", incorrect, "\n")
