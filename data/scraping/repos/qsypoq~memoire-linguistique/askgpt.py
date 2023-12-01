import openai
import sys

openai.api_key = "xxxx"

theme = sys.argv[1]

prompt = f"Ecris moi un texte entre 1000 et 1250 mots qui parle de {theme}"

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo", 
  messages=[{"role": "user", "content": prompt}]
)

with open(f"gpt/txt/{theme}.txt", 'w+') as outfile:
        outfile.write(completion['choices'][0]['message']['content'])
