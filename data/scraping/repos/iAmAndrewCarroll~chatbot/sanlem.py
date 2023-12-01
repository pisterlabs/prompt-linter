import random
import json
import pickle
import numpy as np
import os
import ssl
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model

import openai
from dotenv import load_dotenv
import os

# load env variables from .env
load_dotenv()

# retrieve the API key
api_key = os.getenv("GPT3_API_KEY")
# print("Loaded API key: ", api_key)

nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(nltk_data_path)
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())

words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbot_model.h5")
# model.summary()


def clean_up_sentence(sentence):
    sentence = sentence.lower()
    sentence_words = nltk.word_tokenize(sentence)
    # sentence_words = [lemmatizer.lemmatize(word) for word in sentence]
    return ' '.join(sentence_words)


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    result = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(result) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []

    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    
    # if not return_list:
    #     print("Predicted class: No match")
    # else:
    #     predicted_class = return_list[0]['intent'].lower()
    #     # print("Predicted class: ", predicted_class)

    return return_list

def get_response(intents_list, intents_json):
    user_input = intents_list[0]['user_input'].lower() if 'user_input' in intents_list[0] else None
    list_of_intents = intents_json['intents']

    result = None # default

    # check if user input is not None
    if user_input is not None:
    # check if user input matches patterns intent.json
      for intents in list_of_intents:
          for pattern in intents['patterns']:
              if pattern.lower() in user_input:
                  result = random.choice(intents['responses'])
                  return result
    
    # no matching pattern query GPT3
    if 'user_input' in intents_list[0]:
        user_input = intents_list[0]['user_input']
        response = generate_gpt3_response(user_input)
        return response

    return result if result else ""

def generate_gpt3_response(user_input):
    # print("Inside generate_gpt3_response function")
    # initialize the OpenAI API
    openai.api_key = api_key

    prompt = f"You: {user_input}\nBot: Respond only to the user's query."
    # print("Prompt: ", prompt)

    # generate the response
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=50, # can be adjusted as needed for repsonse length
        n=1, # number of responses to generate
        stop=None, # you can specify stop words to end the response
    )

    # print("GPT3 API Response: ", response)

    if response.choices and response.choices[0].text:
        generated_response = response.choices[0].text
        # print("GPT3 Response: ", generated_response)
        return generated_response
    else:
        # print("GPT3 API did not return a valid response.")
        return ""

def chat():
  print("Go! Bot is running!")

  while True:
      message = input("You: ")
      if message.lower() == "quit":
          print("Bot: Bye!")
          break
      
      # preprocess user input to match training data
      # use the 'punkt' toekinzer to tokenize sentences
      tokenized_message = tokenizer.tokenize(message)
      # join the tokenized sentences into a single string
      cleaned_message = ' '.join(tokenized_message)
      cleaned_message = clean_up_sentence(cleaned_message)
      ints = predict_class(cleaned_message)
      
      res = get_response(ints, intents)

      if res:
          print("Bot: ", res)
      else:
          res = generate_gpt3_response(message)
          if res:
              print("Bot: ", res)
          else:
              print("Bot: ")

if __name__ == "__main__":
    chat()

