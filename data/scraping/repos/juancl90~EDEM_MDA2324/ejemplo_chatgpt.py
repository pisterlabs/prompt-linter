import os
import openai

# Configuramos la API KEY de OPEN AI 
openai.api_key = '' # 'TODO: AQUÍ TIENES QUE PONER TU API KEY DE CHATGPT'

prompt = input('Redacta con detalle el resultado que quieres: ')
file_name = input('Indica el nombre del archivo donde se va a guardar: ')
file_extension = input('Indica la extensión del archivo: .')

# realizamos petición a Open AI
response = openai.Completion.create(
  model="text-davinci-003",
  prompt= prompt,
  temperature=0.7,
  max_tokens=900,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

with open(f'{file_name}.{file_extension}', 'w') as file:
  file.write(response["choices"][0]["text"])