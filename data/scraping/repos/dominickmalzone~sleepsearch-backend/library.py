import pickle
import requests as rq
import pymongo
import os

client = pymongo.MongoClient(os.getenv("mongodb_uri"))
db = client.get_database("queryanswers")
queries = db.query

# print(queries.find_one({'tag' : 'insomnia'})["response"]) # THIS ONE

def database_query_label(sentence) : # general purpose

  classnames = ['adobe creative cloud', 'adobe xd', 'airbnb', 'amazon',
        'amazon prime', 'angular', 'apple', 'apple music', 'apple tv',
        'arduino', 'brain cancer', 'breast cancer', 'c', 'c#', 'c++',
        'cancer', 'cat', 'covid19_summary', 'cryptocurrency', 'css',
        'django', 'dog', 'earthquake', 'excessive daytime sleepiness',
        'express', 'facebook', 'flask', 'github', 'gmail', 'google',
        'google cloud', 'google play', 'hackathon', 'heart cancer', 'html',
        'hulu', 'insomnia', 'instagram', 'java', 'javascript',
        'kidney cancer', 'kotlin', 'laravel', 'linkedin', 'liver cancer',
        'lung cancer', 'lyft', 'microsoft', 'microsoft access',
        'microsoft excel', 'microsoft office', 'microsoft onenote',
        'microsoft outlook', 'microsoft powerpoint', 'microsoft word',
        'minecraft', 'mouse', 'narcolepsy', 'netflix', 'node',
        'objective c', 'parasomnia', 'php', 'python', 'rat', 'react',
        'restless legs syndrome', 'revolutions', 'rich people', 'ruby',
        'shift work disorder', 'slack', 'sleep apnea', 'sleeping',
        'spotify', 'stackoverflow', 'stock market', 'stomach cancer',
        'swift', 'tesla', 'travel', 'trello', 'tsunami', 'twilio',
        'twitter', 'uber', 'valorant', 'volcano', 'vue', 'world war 1',
        'world war 2', 'youtube', 'youtube music']

  model_0 = pickle.load(open("model_0.pkl", 'rb'))
 
  predict =  model_0.predict([sentence])
  proba = model_0.predict_proba([sentence])
  probability = proba[0][predict][0] * 10

  if predict == [77] and probability < 0.27 : 
    result = "0" # its returning 0 bcs the database cannot found the specified query
  else :
    label = classnames[predict[0]]

    if str(label) == "insomnia" or str(label) == "sleep apnea" or str(label) == "sleeping":
      result = "0"
    else :
      try :
        result = queries.find_one({'tag' : str(label) })["response"]
        result = str(result) + " (naive bayes)"
      except : 
        result = "0"
  return result

# print(database_query_label("kidney cancr ")) # the database still can handle typo
# print(database_query_label("stmch cancer "))
# print(database_query_label("what is python"))
import openai
def gpt_three_query(sentence) : 
  # train data 
  train = f"The following is a question answering bot. You can ask it a question about sleep disorders like insomnia or sleep apnea, and it will give you a medically accurate response.\n\nQ: What is the best cpap machine for sleep apnea?\nA: Based on thousands of public user reviews, Resmed Airsense 10 AutoSet is the best for new users. Users appreciate that it's lightweight, has a built-in humidifier, and is easy to use with the companion mobile app. There is a shortage of supplies right now though, due to an increase of demand from a major recall from Philips. So you may find yourself paying extra or searching sites like Secondwindcpap.com for gently used products.\n\n\nQ: How do i treat insomnia due to anxiety\nA: According to the top reddit submissions, there are four top treatments. Practice guided medication, exercise daily, avoid screens late at night, and drinking lemon tea with chamomile before bed. Everyone responds differently so it's recommended you try all of them and be patient while finding what works best!\n\nQ:{sentence}?\n"
  openai.api_key = os.getenv("openai_api_key")
  response = openai.Completion.create(
  engine="davinci",
  prompt= train,
  temperature=0,
  max_tokens=100,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0,
  stop=["\n"]
  ).choices[0].text
  response = response.split()
  response.pop(0)
  response = " ".join(response)

  if response == "Unknown" :
    response = "0"
  else :
    response = str(response) + " (gpt3)"
  return response

def wolfram_query(sentence) : 
  wolfram_id ="WEXTEU-4LRRU6G988"
  wolfram_url = "http://api.wolframalpha.com/v1/result"
  
  params = {
    'appid': wolfram_id,
    'i': sentence,}

  r = rq.get(wolfram_url, params=params)
  response =  r.text
  
  if response == "Wolfram|Alpha did not understand your input" :
    response = "sorry didnt understand ur input"
  else :
    response = response + " (wolfram)"
  return response 

# print(wolfram_query("what long is normal ruler"))
import wikipedia
def wikipedia_query(sentence) : 
  sentence = sentence.split()

  if sentence[0].lower() == "wikipedia" or sentence[0] == "what" : 
    sentence.pop(0)
    sentence = " ".join(sentence)
    try :
      result = wikipedia.summary(sentence, sentences = 2)
    except : 
      result = "0"
  else :
    result = "0"
  return result

def query(sentence): 
  query_label =  database_query_label(sentence)
  wikipedia_response = wikipedia_query(sentence)
  gpt_response = gpt_three_query(sentence)

  if query_label == "0" : 

    if gpt_response == "0" :
      if wikipedia_response == "0" :
        result = wolfram_query(sentence)
      else :
        result = str(wikipedia_response) + " (wikipedia)" 
    else :
      result = gpt_response
  else : 
    # at here we add the mongodb integration
    # that seek our database for answers
    result = query_label

  return result


# query("what is stomach cancer")

# try it right here
# print("~~~")
# print(query("what is insomnia"))
# print(gpt_three_query("How long is a ruler"))

