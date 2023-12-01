import json
import os
import openai
from sqlitedict import SqliteDict
db = SqliteDict('./my_db.sqlite', autocommit=True)

json_file = open("fine_tunes.txt", 'rb').read()
data = json.loads(json_file)['data']
final_item = data[len(data) - 1]
name = final_item['fine_tuned_model']

if name == None:
  exit("error: fine-tune job has failed.")

openai.api_key = os.environ["OPENAI_API_KEY"]


def make_problem(log): 
  prompt = f'remove the timestamp, server name, and any hexadecimal values. {log}'
  response = openai.Completion.create(
      model=name,
      max_tokens=1024,
      temperature=0,
      stop='\n',
      prompt=prompt
  )

  return response['choices'][0]['text']


def process_log(log):
  problem = make_problem(log)

  if problem in db:
    return db[problem]
  else:
    response = openai.Completion.create(
        model=name,
        max_tokens=1024,
        temperature=0.5,
        stop='\n',
        prompt=f"Understand these logs and diagnose a problem and a solution: {log} ->"
    )

    analysis = response['choices'][0]['text']
    db[problem] = analysis
    return analysis

# Random comment

print(process_log(
    'Apr 6 10:30:22 server1 kernel: [ 2601.567890] ata1.00: exception Emask 0x0 SAct 0x0 SErr 0x0 action 0x6'))
print(process_log(
    'Apr 6 10:30:22 server1 kernel: [ 2601.567890] ata1.00: exception Emask 0x0 SAct 0x0 SErr 0x0 action 0x6'))
print(process_log('Apr 6 10:30:22 server1 main.exe: info: connected successfully to db3'))
print(process_log('Apr 6 10:50:32 server1 main.exe: too many requests: put on hold, ten requests timing out'))
