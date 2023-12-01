import openai
import wikipedia
from os import getenv
from dotenv import load_dotenv

load_dotenv()

# pass the api key
openai.api_key = getenv("OPENAI_API_KEY")

# get user input
title = input("Title of the page: ")

# get the wikipedia content
page = wikipedia.page(title=title, auto_suggest=False)

# define the prompt
prompt = "Write a summary of the following article: " + page.content[:10000]
messages = []
messages.append({"role": "user", "content": prompt})

try:
  # make an api call
  response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    temperature=0,
    n=1
  )

  # print the response
  print(response.choices[0].message.content)

# authentication issue
except openai.AuthenticationError as e:
  print("No Valid Token/Authentication Error: %s" % e.message)

# invalid request issue
except openai.BadRequestError as e:
  print("Bad Request Error: %s" % e.message)
