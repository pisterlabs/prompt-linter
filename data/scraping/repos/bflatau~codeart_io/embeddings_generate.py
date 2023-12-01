import os
import openai
import pandas as pd
import re
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

openai.api_key = os.getenv("OPENAI_API_KEY")

ALLOWED_REGEX = r'[a-zA-Z\$\*\%\&\@\!\?\#\'\,\" ]*'

df = pd.read_csv('./embeddings/combined_season1-37.tsv',  sep='\t') #read file
df = df[[ 'answer', 'question']] #name the columns we want
df = df.dropna() #remove all the columns not identified above
#print(len(df))

# Filter out embeddings with characters we can't display
df = df[df.apply(lambda row: (re.fullmatch(ALLOWED_REGEX, row['answer']) is not None) and (re.fullmatch(ALLOWED_REGEX, row['question']) is not None), axis=1)]
print("FILTERED")
#print(df)

df['combined'] = "Question: " + df.question.str.strip() + "; Answer: " + df.answer.str.strip() #make superstring?
#df.head(6) #this tests N rows to see if the data looks correct

#df = df.tail(1000) # set df to the last N number of values to work with
#print(len(df)) #output length


from openai.embeddings_utils import get_embedding

# This will take just under 10 minutes
df['babbage_similarity'] = df.combined.apply(lambda x: get_embedding(x, engine='text-similarity-babbage-001'))
df['babbage_search'] = df.combined.apply(lambda x: get_embedding(x, engine='text-search-babbage-doc-001'))
df.to_csv('embedded_120k_FILTERED_babbage.csv')

print("FINISHED!")