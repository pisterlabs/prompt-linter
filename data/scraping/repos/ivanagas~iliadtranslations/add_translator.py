import json
import sys
from dotenv import load_dotenv
import os
import openai

name = "Michael Reck"

with open('translators.json', 'r+') as file:
  data = json.load(file)
  
  name_key = name.lower().split()[-1]

  # Check if the name already exists in the data
  if name_key in data:
    print(f"Entry for {name} already exists.")
    sys.exit(0)

  # Generate description
  load_dotenv()
  openai.api_key = os.getenv("OPENAI_API_KEY")

  prompt = f"Write a one sentence description of the ${name} translation of the Iliad."

  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": "You are an expert scholar on the Iliad by Homer."},
      {"role": "user", "content": prompt}
    ]
  )

  response_content = response["choices"][0]["message"]["content"]

  # Create the new entry
  new_entry = {
    "translator": name,
    "year": 1900,
     "tags": [],
    "links": {
      "amazon": "",
      "online": ""
    },
    "description": response_content,
    "quotes": {
      "Book 1 - Intro": "",
      "Book 5 - Athena encourages Diomedes": "",
      "Book 6 - Leaves": "",
      "Book 9 - Glory or long life": "",
      "Book 21 - Moaning about death": ""
    }
  }

  # Add the new entry to the data
  data[name_key] = new_entry

  print(f"New entry added for {name}.")

  with open('new_translators.json', 'w') as new_outfile:
    json.dump(data, new_outfile)
