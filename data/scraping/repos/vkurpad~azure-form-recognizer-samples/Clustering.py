

#get_ipython().system('pip install azure-storage-blob')
#get_ipython().system('pip install openai num2words matplotlib plotly scipy scikit-learn pandas tiktoken')


from azure.storage.blob import BlobClient, ContainerClient
import os, logging
import argparse
import openai
import os, io
import re
import requests
import sys
from num2words import num2words
import os
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity
import tiktoken
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score as ss
import json, time
from requests import get, post
import itertools



def read_docs():
    blobs = container_client.list_blobs(prefix)
    for blob in blobs:
        folder, tail = os.path.split(blob.name)
        if(tail[-3:] != "pdf"):
            continue
        url = f"{container_path}/{blob.name}{sas_token}"
        process_file(tail, folder, url)
        
        # if(tail[-4:] != "json"):
        #     continue
        # blob_client = container_client.get_blob_client(blob.name)
        # blob_content = blob_client.download_blob().readall()
        # ocr_res = json.loads(blob_content)
        # content_from_files.append({"fileName":tail, "content": ocr_res["analyzeResult"]["content"]})

def process_file(file_name, file_path, file_url):
    url = v3_url%"prebuilt-layout"
    body = { 
        "urlSource": file_url 
        }
    headers = {
        # Request headers
        'Content-Type': 'application/json',
        'Ocp-Apim-Subscription-Key': apim_key,
    }

    params = {

        "locale": "en-US",
        "pages": "1"
    }
    try:
        resp = post(url = url, data = json.dumps(body), headers = headers, params = params)
        if resp.status_code != 202:
            print("POST analyze failed:\n%s" % resp.text)
            quit()
        get_url = resp.headers["operation-location"]
        res = {
            "file_name": file_name,
            "file_path": file_path,
            "operation_location": get_url
        }
        operation_locations.append(res)
    except Exception as e:
        print("POST analyze failed:\n%s" % str(e))
        quit()

def write_blob(file_name, file_path, contents):
   print("writing blob")
   
   blob_client = container_client.get_blob_client(f"{file_path}/{file_name}.ocr.json")
   blob_client.upload_blob(contents)
       

def get_results(res):
    n_tries = 10
    n_try = 0
    wait_sec = 6
    while n_try < n_tries:
        try:
            resp = get(url = res["operation_location"], headers = {"Ocp-Apim-Subscription-Key": apim_key})
            resp_json = json.loads(resp.text)

            if resp.status_code != 200:
                print("GET Receipt results failed:\n%s" % str(resp.status_code))
                break
            status = resp_json["status"]
            if status == "succeeded":
                content_from_files.append({"fileName": res["file_name"], "content": resp_json["analyzeResult"]["content"]})
                write_blob(res["file_name"], res["file_path"], resp.text)
                return True
            if status == "failed":
                print(f"{model} Analysis failed:\n%s" % resp_json)
                break
            # Analysis still running. Wait and retry.
            time.sleep(wait_sec)
            n_try += 1     
        except Exception as e:
            msg = "GET analyze results failed:\n%s" % str(e)
            print(msg)
            return False

def save_embeddings(df ):
    # Save the enbeddings from the column ada_v2 to azure blob storage

    
    df.to_pickle("embeddings.pkl")
    blob_client = container_client.get_blob_client( f"{prefix}embeddings.pkl")
    with open("embeddings.pkl", "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    os.remove("embeddings.pkl")

def get_scores_and_labels(combinations, X):
    scores  = []
    all_labels_list = []
    for i, (eps, num_samples) in enumerate(combinations):
        dbscan_model = DBSCAN(eps=eps, min_samples=num_samples).fit(X)
        labels = dbscan_model.labels_
        labels_set = set(labels)
        if len(labels < 2):
            continue
        scores.append(ss(X, labels=labels))
        
        all_labels_list.append(labels)
        print(f"Index: {i}, eps: {eps}, num_samples: {num_samples}, score: {scores[i]}")
    if scores == []:
        return None
    best_index = np.argmax(scores)
    best_parameters = combinations[best_index]
    best_labels = all_labels_list[best_index]
    best_score = scores[best_index]
    return {
        "best_index": best_index,
        "best_parameters": best_parameters,
        "best_labels": best_labels,
        "best_score": best_score
    }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fr_endpoint', type=str, help='Form Recognizer endpoint', default='')
    parser.add_argument('--fr_key', type=str, help='key', default='')
    parser.add_argument('--oai_endpoint', type=str, default='')
    parser.add_argument('--oai_key', type=str, default='')
    parser.add_argument('--oai_embeddings_model', type=str, default='')
    parser.add_argument('--container_url', type=str, default='')
    parser.add_argument('--container_sas', type=str, default='')
    parser.add_argument('--folder_prefix', type=str, default='')


    return parser.parse_args()


            
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    print("\n[Form Recognizer] - Docment clustering tool ---------------------")
    print(f"FR Endpoint                            : '{args.fr_endpoint}'")
    print(f"FR key                                 : '{args.fr_key}'")
    print(f"OAI Endpoint                           : '{args.oai_endpoint}'")
    print(f"OAI Key                                : '{args.oai_key}'")
    print(f"OAI embeddings model                   : '{args.oai_embeddings_model}'")
    print(f"Container URL                          : '{args.container_url}'")
    print(f"Container URL                          : '{args.container_sas}'")
    print(f"Folder prefix                          : '{args.folder_prefix}'")

container_client = ContainerClient.from_container_url(f"{args.container_url}{args.container_sas}")
sas_token = args.container_sas
container_path = args.container_url
oai_endpoint = args.oai_endpoint
oai_key = args.oai_key


API_KEY = oai_key
RESOURCE_ENDPOINT = oai_endpoint

openai.api_type = "azure"
openai.api_key = args.oai_key
openai.api_base = args.oai_endpoint
openai.api_version = "2022-12-01"

url = openai.api_base + "/openai/deployments?api-version=2022-12-01" 

#r = requests.get(url, headers={"api-key": API_KEY})

prefix = args.folder_prefix
endpoint = args.fr_endpoint
apim_key = args.fr_key
api_version = "2022-08-31"
v3_url = args.fr_endpoint + f"formrecognizer/documentModels/%s:analyze?api-version={api_version}"
print(v3_url)

operation_locations = []
content_from_files = []

read_docs()

        

while(len(operation_locations) > 0):
    op = operation_locations[0]
    if(get_results(op)):
        operation_locations.pop(0)
            

print(operation_locations)


df = pd.DataFrame.from_dict(content_from_files)


# Since we are only using content from page 1, assume that this will not be needed. 
tokenizer = tiktoken.get_encoding("cl100k_base")
df['n_tokens'] = df["content"].apply(lambda x: len(tokenizer.encode(x)))
df = df[df.n_tokens<8192]
len(df)
blob_client = container_client.get_blob_client( f"{prefix}embeddings.pkl")
if blob_client.exists():
    contents = blob_client.download_blob().readall()
    df_v2 = pd.read_pickle(io.BytesIO(contents))
    embedding_list = df_v2['ada_v2'].to_list()
else:   
    df['ada_v2'] = df["content"].apply(lambda x : get_embedding(x, engine = 'text-embedding-ada-002')) # engine should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model
    save_embeddings(df)
    embedding_list = df['ada_v2'].to_list()

# Compute DBSCAN
# Grid search
eps = [0.3, 0.4, 0.45, 0.55, 0.6]
min_samples = [5, 6, 7, 8, 9, 10, 15]
combinations = list(itertools.product(eps, min_samples))
best_dict = get_scores_and_labels(combinations, embedding_list)
if best_dict is None:
    print("No good parameters found")
else:
    print("Best index: %d" % best_dict["best_index"])
    print("Best parameters: %s" % str(best_dict["best_parameters"]))
    print("Best score: %f" % best_dict["best_score"])

    # Save the labels to the dataframe
    df['labels'] = best_dict["best_labels"]



