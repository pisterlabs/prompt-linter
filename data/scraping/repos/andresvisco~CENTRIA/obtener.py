import os
import openai
openai.api_type = "azure"
openai.api_base = "https://arkano-openai-dev.openai.azure.com/"
openai.api_version = "2022-12-01"
openai.api_key = os.getenv("OPENAI_API_KEY")

def obtener_datos(info_text):
  response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=info_text    ,
    temperature=0.35,
    max_tokens=538,
    top_p=1,
    frequency_penalty=0.27,
    presence_penalty=0,
    best_of=1,
    stop=None)
  return response

def obtener_contrato(text):
  response = openai.Completion.create(
    engine="davinci",
    prompt=text,
    temperature=0.35,
    max_tokens=538,
    top_p=1,
    frequency_penalty=0.27,
    presence_penalty=0,
    best_of=1,
    stop=None)
  return response