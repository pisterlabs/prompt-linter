"""
Author:     Juraj Dedič, xdedic07
Binds to port 5000 and accepts POST requests with JSON body containing the utterance.
It needs OPENAI_API_KEY environment variable and a paid OpenAI account to work.
It could be modified to work with the free OpenAI API, but it would be much slower. 
(free API has 3 requests per minute at the time of writing the thesis)
"""

import openai
import os
import re
import json

print(os.environ['OPENAI_API_KEY'])

# if there's no cache.json, create it
if not os.path.exists("cache.json"):
  cache_file = open("cache.json", "w")
  cache_file.write("[]")
  cache_file.close()

def get_response_chat(utterance):
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": "You detect airplane callsigns. I provide you prompt only containing the sentence, you only ouput one callsign no more words. In case of no callsign you ouput 'None'"},
      # {"role": "user", "content": "sentence: 'ryanair seven three alpha hotel turn left heading three six zero' nearbyCallsigns: 'BLA1RK','BRU861','BRU862','CMB132','CSA2DZ','CSA6KG','CSA918','CSA94D','DEBLN','DLH1935','DLH4RM','DLH6YW','DLH9TP','ECC701','EWG7677','EXS1SG','EZS1217','EZS93AV','KLM44K','MLD864','OKAVK','OKHBT','OKKEA','OKLLZ','OKMHZ','OKOUU37','OKPHM','OKSOL','OKTOP','OKVUS02','OKWUS17','OKYAI14','PGT1527','PGT76A','QTR8004','QTR8005','RYR1JU','RYR3345','RYR73AH','RYR77SJ','RYR7RM','SDR4995','SFS80','SXS7D','THY6286','TIE790J','TVS5RU','UAE139','UAE73','WZZ9654'"},
      # {"role": "assistant", "content": "text: 'ryanair seven three alpha hotel', callsign: 'RYR73AH'"},
      {"role": "user", "content": "ryanair seven three alpha hotel turn left heading three six zero"},
      {"role": "assistant", "content": "ryanair seven three alpha hotel"},
      {"role": "user", "content": "cleared for low pass runway two four oscar kilo bravo alpha lima"},
      {"role": "assistant", "content": "oscar kilo bravo alpha lima"},
      # {"role": "user", "content": "Skytravel Two Eight Two Seven and there is closed echo airspace below flight level one hundred so if you wish to avoid class echo airspace next descend in two zero miles"}
      {"role": "user", "content": utterance}
    ]
  )
  return response

def get_response_text(utterance):

  context = """CallSignAI is a system that detects airplane callsigns. I provide you prompt only containing the sentence and nearby callsigns, you only ouput one callsign no more words. In case of no callsign you ouput 'None'

User: ryanair seven three alpha hotel turn left heading three six zero
CallsignAI: ryanair seven three alpha hotel
User: cleared for low pass runway two four oscar kilo bravo alpha lima
CallsignAI: oscar kilo bravo alpha lima
User:"""

  response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=context+utterance,
    # temperature=0.9,

  )

  return response

# normalize the text for uniformity
def normalize_text(txt):
    txt = txt.lower()
    txt = txt.replace('<eps>', '') \
             .replace('alfa', 'alpha') \
             .replace('niner', 'nine')
             
    txt = re.sub(r'\bjuliet\b', 'juliett', txt)
    txt = re.sub(r'_([a-z])_', r'\1', txt)
    txt = txt.replace('_', '')
    return txt


# print(get_response_text("Swiss One Three Eight Apron guten tag push startup approved"))

def get_answer(utterance):
  
  response = get_response_chat(utterance)
  print(response)

  answer = response.choices[0].message.content

  # if the last character is a dot, remove it
  if answer[-1] == ".":
    answer = answer[:-1]

  answer_lower = answer.lower()
  answer_lower_tokens = answer_lower.split(" ")

  if answer_lower == "none" or answer_lower_tokens[0] == "none" or answer_lower_tokens[0] == "none.":
    return None
  return answer
  # return response

# implemented simple cache for saving money and time when experimenting

def get_answer_from_cache(utterance):
  """
  Returns the answer from cache.json if it exists, otherwise returns None
  """
  cache = open("cache.json", "r")
  try:
    cache_json = json.load(cache)
  except:
    cache_json = []

  for entry in cache_json:
    if entry["utterance"] == utterance:
      cache.close()
      return entry["answer"]
  return None

def add_to_cache(utterance, answer):
  """
  Adds the utterance and answer to cache.json
  """
  try:
    cache_file = open("cache.json", "r")
    cache_json = json.load(cache_file)
    cache_file.close()
  except:
    cache_json = []

  # add to cache.json
  cache_json.append({
    "utterance": utterance,
    "answer": answer
  })
  
  # write cache.json
  cache_file = open("cache.json", "w")
  json.dump(cache_json, cache_file)
  cache_file.close()


from flask import Flask, request, jsonify

app = Flask(__name__)

# route for getting the responses for callsign detection task
@app.route("/", methods=['POST'])
def hello_world():
  content_type = request.headers.get('Content-Type')
  if (content_type == 'application/json'):
    json = request.json

    utterance = json['text']
    # lower case
    utterance = utterance.lower()

    # check cache
    cached_answer = get_answer_from_cache(utterance)

    if cached_answer != None:
      answer = cached_answer
      print("answer from cache")
    else:
      answer = get_answer(utterance)
      # add to cache
      add_to_cache(utterance, answer)


    print("text:",utterance)
    print("answer:",answer)

    json = {
      "text": answer,
      "span": answer
    }

    results_final = []
    results_final.append(json)
    return jsonify(results_final)
  else:
    return 'Content-Type not supported!'
    
app.run(port = 5000)