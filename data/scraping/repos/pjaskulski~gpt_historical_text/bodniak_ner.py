""" openai test - extraction information from text
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import openai


env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)

OPENAI_ORG_ID = os.environ.get('OPENAI_ORG_ID')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

openai.api_key = OPENAI_API_KEY

# dane z pliku tekstowego
file_data = Path("..") / "data" / "bodniak.txt"
with open(file_data, 'r', encoding='utf-8') as f:
    data = f.read()

prompt = f"W podanym tekście wyszukaj informacje o osobach, znalezione osoby " \
          "wypisz w formie listy, jeżeli są znane funkcje lub urzędy osób " \
          "umieść te informacje w nawiasach obok osób: \n\n" +  data

prompt = f"W podanym tekście wyszukaj słowa kluczowe będące nazwami własnymi, znalezione słowa kluczowe " \
         "wypisz w formie listy, w nawiasach obok słowa kluczowego umieść informację " \
         "o typie słowa kluczowego np. osoba, miejsce, rzeka, morze.\n\n" + data

response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    temperature=0.0,
    max_tokens=500,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
)

print(response['choices'][0]['text'])