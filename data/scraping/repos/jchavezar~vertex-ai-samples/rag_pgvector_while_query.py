#%%
import csv
import os
import base64
import time
import asyncio
from k import k
import pandas as pd
import streamlit as st 
from utils import google
from openai import OpenAI
from utils.credentials import *

os.environ["OPENAI_API_KEY"] = k

variables = {
    "project_id": "vtxdemos",
    "region": "us-central1",
    "instance_name": "pg15-pgvector-demo",
    "database_user": "emb-admin",
    "database_password": DATABASE_PASSWORD,
    "database_name": "rag-pgvector-langchain-1",
    "docai_processor_id": "projects/254356041555/locations/us/processors/7e6b9d94d3bafa4f",
    "location": "us"
}

parameters = {
    "candidate_count": 1,
    "max_output_tokens": 1024,
    "temperature": 0.2,
    "top_p": 0.8,
    "top_k": 40
}

client = google.Client(variables)

async def db_functions(documents, query):
    await client.create_table()
    await client.insert_documents_vdb(documents)    
    return await client.query(query)

# %%
# LLM prompt + context
documents, ocr_time, embeddings_time = client.prepare_file("../documentai/1065.pdf")
#st.write(f"Embeddings time: {round(embeddings_time, 2)} sec")
start = time.time()

#%%
await client.create_table()
await client.insert_documents_vdb(documents)

def open_ai_chatpgt(prompt, context):
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=k,
    )

    completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"""

                Prompt: {prompt}

                Context: {context}

                """,
            }
        ],
        model="gpt-4",
    )
    return completion.choices[0].message.content

#%%
responses = []
with open("questions.csv", "r") as f:
    reader = csv.reader(f)
    for n,row in enumerate(reader):
        if n%10 == 0:
            time.sleep(60)
        matches = await client.query(row[0])
        matches = pd.DataFrame(matches)
        responses.append({
            "query": row[0], 
            "text-bison-32k": client.llm_predict(row[0], context=pd.DataFrame(matches).to_json(), parameters=parameters),
            "chatgpt-4": open_ai_chatpgt(row[0], context=pd.DataFrame(matches).to_json())            
            })
        

# %%

keys = responses[0].keys()

with open("llm_responses.csv", "w", newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(responses)
    
#%%    
matches = await client.query("what is the gross profit ?")
# %%
