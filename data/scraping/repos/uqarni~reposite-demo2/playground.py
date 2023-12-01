from functions import find_examples, ideator, initial_text_info
import openai
import pandas
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pandas as pd
from playground2 import generate_response
from datetime import datetime
import time
#open permutation_test in pandas
df = pd.read_csv('permutation_test.csv')
df['RAG 50 Response'] = df['RAG 50 Response'].astype(object)

# Create the output CSV and write the header
output_path = 'permutation_test_modified.csv'
df.iloc[0:0].to_csv(output_path, index=False)

# Processing
now = datetime.now()
print('start time: ' + str(now))

for index, row in df.iterrows():
    if pd.notna(row['User Message Reworded']):
        processed_value = generate_response(row['User Message Reworded'])
        print(processed_value)
        df.at[index, 'RAG 50 Response'] = processed_value
        time.sleep(10)
        
        # Append the updated row to CSV
        df.iloc[index:index+1].to_csv(output_path, mode='a', header=False, index=False)

now = datetime.now()
print('end time: ' + str(now))
