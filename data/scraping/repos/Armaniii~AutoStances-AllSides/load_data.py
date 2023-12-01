# import chromadb
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import preprocessing as preprocessing
import os
import langchain
import openai
import sys
from time import sleep
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA, RetrievalQA
from langchain.retrievers.self_query.base import SelfQueryRetriever
from chromadb.utils import embedding_functions


# import chromadb
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import preprocessing as preprocessing
import os
import langchain
import openai
import sys
from time import sleep
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA, RetrievalQA
from langchain.retrievers.self_query.base import SelfQueryRetriever
from chromadb.utils import embedding_functions
import chromadb 


os.environ["OPENAI_API_KEY"] = ""

openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key = os.environ.get("OPENAI_API_KEY"),model_name="text-embedding-ada-002")

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import CSVLoader


persist_directory = 'C:/Users/arman/Workspace/phd/Arman/autostances/data/chroma_database/'

def delete_reddit():
  db_client = chromadb.PersistentClient(path=persist_directory)
  db_client.delete_collection("reddit")


def upload_reddit_data(data):
  db_client = chromadb.PersistentClient(path=persist_directory)
  reddit_collection = db_client.get_or_create_collection("reddit_v2", embedding_function=openai_ef)
  print(db_client.list_collections())

  # check chroma collection embedding function

  # There is an issue here with uploading the documents

  # print(reddit_collection._embedding_function)
  # print(stop)

  batch_size = 100  # how many embeddings we create and insert at once
  # remove none values from dataframe
  data = data[data['comment'] != None]

  for i in tqdm(range(0, len(data), batch_size)):
    # find end of batch
    i_end = min(len(data), i + batch_size)
    meta_batch = data[i:i_end]

    preprocessing.clean_master(meta_batch)
    meta_batch = meta_batch.fillna(0)
    # get ids
    ids_batch = list(map(str, meta_batch.index.to_list()))

    texts = meta_batch["comment"].to_list()
   

    meta_batch1 = {
      'author': meta_batch['author'].to_list(),
      'comment_length': meta_batch['comment_length'].to_list(),
      'parent_id': meta_batch['parent_id'].to_list(),
      'is_op': meta_batch['is_op'].to_list(),
      'score': meta_batch['score'].to_list(),
      'forum': meta_batch['forum'].to_list(),
      'title': meta_batch['title'].to_list(),
      'date': meta_batch['timestamp'].to_list()
    }

    meta_batch1 = meta_batch.to_dict('records')

    # upsert to CHROMA

    reddit_collection.upsert(ids=ids_batch, documents=texts, metadatas=meta_batch1)

def upload_congress_data(data):
  db_client = chromadb.PersistentClient(path=persist_directory)
  congress_collection = db_client.get_or_create_collection("congress", embedding_function=openai_ef)
  print(db_client.list_collections())

  batch_size = 100  # how many embeddings we create and insert at once
  # remove none values from dataframe
  data = data[data['speech2'] != None]

  for i in tqdm(range(0, len(data), batch_size)):
    # find end of batch
    i_end = min(len(data), i + batch_size)
    meta_batch = data[i:i_end]

    # preprocessing.clean_master(meta_batch)
    meta_batch = meta_batch.fillna(0)
    # get ids
    ids_batch = list(map(str, meta_batch.index.to_list()))

    texts = meta_batch["speech2"].to_list()
   

    meta_batch1 = {
      'congress': meta_batch['chamber'].to_list(),
      'committee_name': meta_batch['committee_name'].to_list(),
      'committee_code': meta_batch['committee_code'].to_list(),
      'title': meta_batch['title'].to_list(),
      'govtrack': meta_batch['govtrack'].to_list(),
      'ranking': meta_batch['ranking'].to_list(),
      'speaker_last': meta_batch['speaker_last'].to_list(),
      'speaker_first': meta_batch['speaker_first'].to_list()

    }
    meta_batch1 = meta_batch.to_dict('records')

    # upsert to CHROMA

    congress_collection.upsert(ids=ids_batch, documents=texts, metadatas=meta_batch1)

def read_reddit_data():
  # Read in Reddit Data
  data = pd.read_csv('C:/Users/arman/Workspace/phd/Arman/autostances/data/gcp_reddit_data.csv')
  data = data.drop(columns=['Unnamed: 0'])

  # convert int columns to strings because we need lists of strings
  data["score"] = data["score"].astype(str)
  data["comment_length"] = data["comment_length"].astype(str)

  #convert boolean values to strings values
  data["is_op"] = data["is_op"].astype(str)

  # replace any nan rows with "None"
  data = data.replace(np.nan, "None")
  return data

def read_congress_data():
  data = pd.read_csv('C:/Users/arman/Workspace/phd/Arman/autostances/data/hearing_GMO.csv')

  # convert int columns to strings because we need lists of strings
  data["govtrack"] = data["govtrack"].astype(str)
  data["congress"] = data["congress"].astype(str)
  data["ranking"] = data["ranking"].astype(str)
  # data["text"] = data["speech"] + data["speech2"]
  # replace any nan rows with "None"
  data = data.replace(np.nan, "None")
  return data


if __name__ == "__main__":
  # delete_reddit()
  data = read_reddit_data()
  upload_reddit_data(data)
  # data = read_congress_data()
  # upload_congress_data(data)