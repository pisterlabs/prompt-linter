#CREATING A EMBEDDING FOR A TEXT AND STORING IT IN A CSV FILE
import pandas as pd
import openai
from openai.embeddings_utils import get_embedding
from transformers import GPT2TokenizerFast
import os
from dotenv import load_dotenv

load_dotenv('.env')

#definir la clé OpenAI
openai.api_key = os.environ.get("OPENAI_API_KEY")
 
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


#Ouvrir un fichier csv et le lire
print("open file test_modified-1.csv")
input_datapath = 'test_modified-1.csv' 
# Lire le fichier CSV en utilisant le séparateur ';' et en ignorant les lignes incorrectes
df_full = pd.read_csv(input_datapath, sep=';', error_bad_lines=False, encoding='utf-8')
print("file loaded")


#renommer les colonnes du dataframe
df = df_full[['structure']]
print("df loaded with " + str(len(df)) + " rows")

#supprimer les lignes avec des valeurs manquantes 
df = df.dropna()
print("df cleaned with " + str(len(df)) + " rows")

#df['combined'] = df.prompt.str.strip() + "\n\n" + df.completion.str.strip()

# supprimer les enregistrements trop longs > 8000 tokens
df['n_tokens'] = df.structure.apply(lambda x: len(tokenizer.encode(x)))
df = df[df.n_tokens<8000].tail(2000)
len(df)
print("df cleaned with " + str(len(df)) + " rows") 


# Aprés 5 à 10 minutes
def get_embedding(text, engine="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   print("get embedding for " + text)
   return openai.Embedding.create(input = [text], model=engine)['data'][0]['embedding']
 
df['ada_embedding'] = df.structure.apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
print("embeddings created")
# ecrire la valeur du vecteur contenue dans le dataframe dans un second CSV
df.to_csv('testembeddings.csv', index=False)
#print(df[['prompt','completion','ada_embedding']])

