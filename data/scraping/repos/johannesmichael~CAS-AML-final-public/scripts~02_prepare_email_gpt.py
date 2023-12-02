import os
import re
from io import BytesIO
import openai

from azure.data.tables import TableServiceClient
from azure.storage.blob import BlobServiceClient
from datetime import date
from tqdm import tqdm
from numpy import  array_split

import tiktoken
from timeit import default_timer

from pandas import DataFrame, to_datetime

import quopri


START_TIME = None
OUTLOOK_CONTENT_CONNECTION_STRING = os.environ.get('OUTLOOK_CONTENT_CONNECTION_STRING')




#load data from azure storage table and create data frame

def load_data():
    # Create the TableServiceClient object which will be used to create a container client
    connect_str = OUTLOOK_CONTENT_CONNECTION_STRING
    table_service = TableServiceClient.from_connection_string(connect_str)
    table_name = "outlookjohannes"
    table_client = table_service.get_table_client(table_name) 
    documents = []
    for entity in table_client.list_entities():
        documents.append(entity)
    df =DataFrame(documents)
    
    return df


#clean out content
def clean_content(row):
    content = row['content']
    #content = content.replace("\r\n", "\r")
    content = content.lstrip('>')
    content = re.sub(r'\*{2,}', '',content)
    content = re.sub(r"\[(.*?)\]", " ", content)
    content = re.sub(r"[^\x00-\x7Füöä]+", " ", content)
    content = re.sub(r"_{3,}", " ", content)

    return content

#decode subject if encoded with quopri by checking the encoding is the beginning of the string
def decode_subject(value):
    if value.startswith("=?"):
        #extract encoding
        encoding = value.split("?")[1]
        subject = quopri.decodestring(value).decode(encoding)
        #subject = value.decode(encoding)
        #remove encoding from subject
        subject = subject.split("?")[3]
    else:
        subject = value
    return subject


#unction to count tokens
def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    #set encoding for openai
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    #encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens



def clean_data(df):
    
    #groupeby df by conversation id 
    #drop all in each group except the one with the youngest received_datetime

    # Ensure 'timestamp' column is in datetime format
    df['received_datetime'] = to_datetime(df['received_datetime'])

    #drop rows with empty content
    df = df[df['content'].notna()]

    # Group by 'conversation_id' and find the row with the maximum 'timestamp'
    idx = df.groupby('conversation_id')['received_datetime'].idxmax()

    # Use the indices of the rows with the maximum 'timestamp' to create a new DataFrame
    df_latest = df.loc[idx]
    df_latest.reset_index(drop=True, inplace=True)
    
    #df_latest = df_latest[~df_latest['PartitionKey'].isin(drop_list_PartitionKey)]
    df_latest.reset_index(drop=True, inplace=True)
    df_latest['content_cleaned'] = df_latest.apply(clean_content, axis=1)
    df_latest['subject'] = df_latest['subject'].apply(decode_subject)
    df_latest["content_length"] = df_latest["content_cleaned"].apply(lambda x: len(x))
    df_latest["content_tt_token_lenght"] = df_latest["content"].apply(lambda x: num_tokens_from_string(x))



    return df_latest

#funciton to query chatgpt with content, ask for classification and return response
def get_summary(value):
    count = num_tokens_from_string(str(value))
    if count < 500:
        max_tokens = 200
    else:
        max_tokens = count/2
    prompt = f"""Erstelle eine Zusammenfassung der folgenden Email-Unterhaltung in <>, inklusive der Personen, die daran beteiligt sind.
                Beispiel:
                Personen: 
                Zusammenfassung: 
                <{value}>
                 """
    
    messages = [{"role": "user", "content": prompt}]
    #check if content is longer than 3000 tokens, if yes use 16k model
    if count < 3000:
        model = "gpt-3.5-turbo"
        max_tokens= max_tokens
    else:
        model = "gpt-3.5-turbo-16k"
        max_tokens=count/4


    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0, # this is the degree of randomness of the model's output
            max_tokens=max_tokens, # this is the maximum number of tokens that the model will generate
            n=1, # this is the number of samples to return
        )
    except:
        response = {"choices": [{"finish_reason": "Error"}]}
    return response


#funciton to query chatgpt with content, ask for classification and return response
def get_completion(value):
    
    prompt = f"""
                Analysiere folgende Email-Unterhaltung, getrennt durch <>, nach folgenden Kriterien:
                - Sender
                - Gesendet
                - Betreff
                - Nachricht (nur Text, entferne Signaturen, Adressen, Bilder, Links, Disclaimer und Fussnoten)
                - Typ (Frage, Antwort, Information, Aufforderung, Werbung...)

                Antwort in einer Liste. Einträge getrennt durch <br>. Beispiel:
                
                <br>
                Typ: 
                Sender: 
                Gesendet: 
                Subject: 
                Nachricht:
                <br>
                <{value}>
                """
    
    count = num_tokens_from_string(value)
    #check if content is longer than 2000 tokens, if yes use 16k model
    # 2000 tokens since the prompt is longer
    if count < 2000:
        model = "gpt-3.5-turbo"
        max_tokens=3800 - count
    else:
        model = "gpt-3.5-turbo-16k"
        max_tokens=15500 - count
    messages = [{"role": "user", "content": prompt}]
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0, # this is the degree of randomness of the model's output
            max_tokens=max_tokens, # this is the maximum number of tokens that the model will generate
            n=1, # this is the number of samples to return
        )
        return response
    except:
        response = {"choices": [{"finish_reason": "Error"}]}
        return response
    


#function to upload data to azure blob storage
def upload_data(df, suffix):
    #get today's date
    today = date.today().strftime('%Y-%m-%d')
    try:
        #Save to Azure Blob Storage
        # Create the BlobServiceClient object which will be used
        blob_service_client = BlobServiceClient.from_connection_string(OUTLOOK_CONTENT_CONNECTION_STRING)

        container_name = 'outlookcontent'
        
        # Create a blob client using the local file name as the name for the blob
        file_name = today + "_outlook_" + suffix + ".parquet"
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=file_name)
        
        # save dataframe to csv
        #csv_file = df.to_csv(index=False)

        parquet_file = BytesIO()
        df.to_parquet(parquet_file,  engine='pyarrow')
        parquet_file.seek(0)  # change the stream position back to the beginning after writing
        response = blob_client.upload_blob(data=parquet_file, overwrite=True)

        
    except:
        df.to_parquet(today + "_finetune.parquet", engine='pyarrow')
    else:
        return response

#function to apply get_completion function and get_summary to each row
def add_columns(df):
    
    df["finish_reason"] = df['content_processed'].apply(lambda x: x["choices"][0]["finish_reason"])
    df = df[df['finish_reason'] != 'Error']
    df["content_string"] = df['content_processed'].apply(lambda x: x["choices"][0]["message"]["content"])
    
    return df


tqdm.pandas()

if __name__ == "__main__":
    #load data
    df = load_data()
    print(df.shape)
    
    START_TIME = default_timer()
    df = clean_data(df)
    print(df.shape)
    #for testing
    #df = df[:3].copy()
    print(df.shape)

    # Multiprocessing speeds up the process but the API call limit seems to be a problem, lots of errors
    
    # Using multiprocessing with 4 processes to apply get_completion function to each row
    #with ThreadPoolExecutor(max_workers=4) as executor:
    #    df['content_processed'] = list(tqdm(executor.map(get_completion, df['content']), total=len(df)))
    df['content_processed'] = df['content'].progress_apply(get_completion)
    upload_data(df, suffix="processed_1000b")

    df = add_columns(df)
    print(df.shape)
    # Using multiprocessing with 4 processes to apply get_summary function to each row
    #with ThreadPoolExecutor(max_workers=4) as executor:
    #    df['content_summary'] = list(tqdm(executor.map(get_summary, df['content_string']), total=len(df)))
    df['content_summary'] = df['content_string'].progress_apply(get_summary)
    upload_data(df, suffix="summary_1000b")

    elapsed_time = default_timer() - START_TIME
    completed_at = "{:5.2f}s".format(elapsed_time)
    print(f"completed in {completed_at}")
 
