from pprint import pprint
import jsonlines
import pandas as pd
from dotenv import load_dotenv
import openai
import os
import time
from datetime import datetime
from tqdm import tqdm

# Carregando o .env e atribuindo a variável de ambiente
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Carregando a base de dados criada
df = pd.read_csv("dataset/output/dataset.csv")
if df['chunk'].isnull().sum() == 0:
    print("All chunks are filled\n")

# Definido a estrutura do prompt template
prompt_template = """Sua Tarefa é resumir ao máximo o Texto fornecido, porém mantendo os principais pontos. 

### Texto:
{input}

### Resumo:"""


# Definindo o caminho para salvar os arquivos

path = '.\\processed_dataset\\processed_input\\'
path_out = '.\\processed_dataset\\processed_output\\'


# Execução do pipeline -> Criando os arquivos de input e output

for i in tqdm(range(0, len(df['chunk']))):
        
        chunk_name = "{}_chunk-{}".format(df['file_name'][i], i).replace(".txt", "") + ".txt"

        if chunk_name in os.listdir(path):
                                
                print("Documento {} já foi processado!".format(chunk_name))

        else:

                completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                        {"role": "system", "content": ""},
                        {"role": "user", "content": prompt_template.format(input=df['chunk'][i])},
                ],
                max_tokens = 256,
                temperature = 0
                        )

                with open(path + chunk_name, 'w', encoding='utf-8') as f:
                                f.write(prompt_template.format(input=df['chunk'][i]))


                resposta = completion['choices'][0]['message']['content']

                with open(path_out + chunk_name, 'w', encoding='utf-8') as f:
                                f.write(resposta)

                print("A taxa de compressão do chunk {} foi de {} %".format(i, round((1-(len(resposta)/len(df['chunk'][i])))*100,2)))


