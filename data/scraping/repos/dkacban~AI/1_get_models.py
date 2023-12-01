import os
import openai
openai.organization = "org-UpMJfYAwK3diGzF1OVSVLb1e"
openai.api_key = os.getenv("OPENAI_API_KEY")
models = openai.Model.list()

for model in models['data']:
    print(model)

# Zadanie: Wyświetl wszystkie informacje o każdym modelu
# Zadanie: Sprawdź dostępne modele zaczynające się od słów gpt-3.5-turbo
