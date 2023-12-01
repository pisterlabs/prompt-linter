# from qa.bot import GroundedQaBot
# import requests
# import json
# import os
import os
import cohere
# from qdrant_client.http import models
# from qdrant_client.http.models import PointStruct
co = cohere.Client(os.environ['mavqa'])



# cohere_api_key = os.environ['mavqa']
# serp_api_key = os.environ['serp']
# rapidapi_key = os.environ['rapid']

# bot = GroundedQaBot(cohere_api_key, serp_api_key)

# def get_language(payload, api_key):
#   url = "https://translate-plus.p.rapidapi.com/language_detect"

#   payload = {"text": payload}
#   headers = {
#     "content-type": "application/json",
#     "X-RapidAPI-Key": api_key,
#     "X-RapidAPI-Host": "translate-plus.p.rapidapi.com"
#   }

#   response = requests.request("POST", url, json=payload, headers=headers)
#   return json.loads(response.text)["language_detection"]["language"]

# #translate
# def translate(payload, api_key, detected_lang='en', to_en=True):
#   if to_en:
#     payload = {
#       "langpair": f"{detected_lang}|en",
#       "q": payload,
#       "mt": "1",
#       "onlyprivate": "0",
#       "de": "a@b.c"
#     }
#   else:
#     payload = {
#       "langpair": f"en|{detected_lang}",
#       "q": payload,
#       "mt": "1",
#       "onlyprivate": "0",
#       "de": "a@b.c"
#     }
#   url = "https://translated-mymemory---translation-memory.p.rapidapi.com/get"
#   headers = {
#     "X-RapidAPI-Key": rapidapi_key,
#     "X-RapidAPI-Host":
#     "translated-mymemory---translation-memory.p.rapidapi.com"
#   }

#   response = requests.request("GET", url, headers=headers, params=payload)

#   return json.loads(response.text)["responseData"]["translatedText"]

# def MaverickQA(query):
#   bot = GroundedQaBot(cohere_api_key, serp_api_key)
#   verbosity = 0
#   print("querying")
#   question = query
#   detected_lang = get_language(query, rapidapi_key)
#   if detected_lang != 'en':
#     question = translate(query, rapidapi_key, detected_lang, True)
#     print(question)
#   reply, source_urls, source_texts = bot.answer(question,
#                                                 verbosity=verbosity,
#                                                 n_paragraphs=5)

#   print(reply)
#   if detected_lang != 'en':
#     reply = translate(reply, rapidapi_key, detected_lang, False)
#   # sources_str = "\n".join(list(set(source_urls)))
#   # reply_incl_sources = f"{reply}\nSource:\n{sources_str}"
#   print(reply)
#   return reply


def MaverickQA(query):
  response = co.embed(texts=[query], model="large")
  embeddings = response.embeddings[0]
  embeddings = [float(i) for i in embeddings]

  search_result = qdrant_client.search(collection_name="mycollection",
                                       query_vector=embeddings,
                                       limit=5)
  # print(search_result)
  # print(search_result)

  # prompt = "Answer the question given the following context:\n"
  # for result in search_result:
  #     for i in result.payload['text']:
  #         prompt += i
  #     prompt += "\n---\n"
  # prompt += "Question:" + query + "\n---\n" + "Answer:"

  #changed the prompt to better reflect the question

  prompt = "This is an example of question answering based on a text passage:\nContext:  "
  for result in search_result:
    for i in result.payload['text']:
      prompt += i
    prompt += "\n---\n"
  prompt += "Question:" + query + "\n---\n" + "Answer:"

  print(prompt)

  print("----PROMPT START----")
  print(":", prompt)
  print("----PROMPT END----")

  response = co.generate(model='command-xlarge-nightly',
                         prompt=prompt,
                         max_tokens=300,
                         temperature=0.9,
                         k=0,
                         p=0.75,
                         stop_sequences=['---'],
                         return_likelihoods='NONE',
                         logit_bias={'11': -10},
                         num_generations=5)

  return response.generations[0].text
