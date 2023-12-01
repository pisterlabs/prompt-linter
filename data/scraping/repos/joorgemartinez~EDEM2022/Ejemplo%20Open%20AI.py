import os
import openai
from dotenv import load_dotenv


#Configuramos la API KEY de OPENAI
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


prompt = input('Redacta con detalle el resultado que quieres: \n')
file_name = input ('Indica el nombre del archivo donde se va a guardar: \n')
file_extension = input('Indica la extensión del archivo: .')


#Realizamos petición a OPEN AI
response = openai.Completion.create(
  model="text-davinci-003",
  prompt=prompt,
  temperature=0.7,
  max_tokens=922,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)


with open(f'{file_name}.{file_extension}', 'w') as file: 
    file.write(response['choices'][0]['text'])
