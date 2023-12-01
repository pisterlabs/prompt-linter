import os
import json
import pickle

import tools
from .general import read_dictionary, jprint
from models import QueryFile, GResult

import sqlalchemy
from sqlalchemy import select
import weaviate
import openai

from flask import g
#from app import app



def create_data_bundle_weaviate(queryPK, db, export = False):
    '''
    Creates a dictionary of (data_obj, embeddings) to be uploaded to Weaviate. 
    Each document is chunkified, each chunk is embedded via OpenAI.
    data_object matches a predefined weaviate schema.

    Input : queryPK
    Output : dictionary - key : (pk, order) value : (data_object, embedding)

    '''

    '''
    Parameters
    '''
    settings_dict = read_dictionary('settings.json')
    openai.api_key = settings_dict["OpenAI_KEY"]

    MODEL = 'text-embedding-ada-002'
    chunk_limit = 1000
    chunk_overlap = 100

    stmt = select(GResult).where(GResult.queryPK == queryPK)
    results = db.session.execute(stmt)

    object_dict = {}

    for i, result in enumerate(results):
        if i > 20 : break

        row = result[0]
        pk = row.pk
        text = row.text

        chunks = tools.chunkify(text, chunk_limit, chunk_overlap, stats = False)

        result = openai.Embedding.create(input = chunks, model = MODEL)

        data_list = result['data']

        for data in data_list :
            order = data['index']
            chunk = chunks[order]
            embedding = data['embedding']

            data_object = {
                'text' : chunk,
                'QueryPK' : queryPK,
                'DocumentPK' : pk,
                'Order' : order
            }

            object_dict[(pk, order)] = data_object, embedding

    if export == True:
        with open('data/openai/'+str(queryPK)+'data_object.pkl', 'wb') as file :
            pickle.dump(object_dict, file)

    return object_dict


def upload_data_weaviate(bundle):
    '''
    Upload data bundle to Weaviate db. 

    Input : 
    bundle : data object from create_data_bundle_Weaviate

    Return : 
    None

    '''
    class_name = 'Text_chunk'
    settings_dict = read_dictionary('settings.json')



    client = weaviate.Client(
    url = "http://localhost:8080",  # Replace with your endpoint
    
    additional_headers = {
        "X-OpenAI-Api-Key" : settings_dict["OpenAI_KEY"]
    }
    )
    
    print('Before upload : ')
    tools.jprint(client.query.aggregate(class_name).with_meta_count().do())

    with client.batch as batch :
        batch.batch_size = 50
        batch.dynamic = True

        for data_object, embedding in bundle.values() :

            batch.add_data_object(
                data_object,
                class_name,
                vector = embedding
            )

    print('After upload : ')
    tools.jprint(client.query.aggregate(class_name).with_meta_count().do())
    
    del client
    return



def oai_embedding(user_chat):
    '''
    Get embedding of user chat input from OpenAI

    Input :
    user_chat : Input from user

    Output :
    vector : Vector from OpenAI to be used for vector search.

    '''
    settings_dict = read_dictionary('settings.json')
    openai.api_key = settings_dict["OpenAI_KEY"]


    MODEL = 'text-embedding-ada-002'
    oai_bundle = openai.Embedding.create(input = [user_chat], model = MODEL)
    oai_vector = oai_bundle['data'][0]['embedding']

    return oai_vector


def query_weaviate(vector, n : int = 5):
    '''
    Vector search in Weaviate db.

    Input : 
    vector : embedding of user input - from OpenAI.

    Output :
    results : results dictionary from Weaviate
    '''

    class_name = 'Text_chunk'
    MODEL = 'text-embedding-ada-002'
    settings_dict = read_dictionary('settings.json')

    client = weaviate.Client(
        url = "http://localhost:8080",
        additional_headers = {"X-OpenAI-Api-Key" : settings_dict["OpenAI_KEY"]}
    )

    results = (client.query.get(class_name, ['text', 'documentPK']
    ).with_near_vector({
        'vector' : vector
    }).with_limit(n)
    .with_additional(['certainty'])
    .do())

    ## need to validate

    texts = []
    dpks = []
    sims = []
    for result in results['data']['Get']['Text_chunk']:

        #SQL query for url  given DPK
        texts.append(result['text'])
        dpks.append(result['documentPK'])
        sims.append(result['_additional']['certainty'])

    return texts, dpks, sims

def chat_response(user_chat, query_results):
    '''
    Sends user input and relevant context(s) to GPT in a prompt and handles response.

    Input :
    user_chat : User input
    query_results : Results dictionary from Weaviate query

    Output :
    bundle : Bundle that contains response from GPT as well as other information presented to user.
    '''
    settings_dict = read_dictionary('settings.json')
    openai.api_key = settings_dict["OpenAI_KEY"]
    
    contexts = ''

    print(query_results)

    for text, link, sim in query_results:
        contexts += f'Context : {text}\n'


    try : 
        user_content = f'''Use the contexts provided to answer the question.''' + contexts + f'''Question : {user_chat}'''
        gpt_response = openai.ChatCompletion.create(
            model = 'gpt-3.5-turbo',
            messages = [
                {'role' : 'system', 'content' : '''You are a helpful assistant that answers questions. Do not guess if you don't know.'''},
                {'role' : 'user', 'content' : user_content}
            ]
        )

    except openai.error.Timeout as e:
      #Handle timeout error, e.g. retry or log
      print(f"OpenAI API request timed out: {e}")
      pass
    except openai.error.APIError as e:
      #Handle API error, e.g. retry or log
      print(f"OpenAI API returned an API Error: {e}")
      pass
    except openai.error.APIConnectionError as e:
      #Handle connection error, e.g. check network or log
      print(f"OpenAI API request failed to connect: {e}")
      pass
    except openai.error.InvalidRequestError as e:
      #Handle invalid request error, e.g. validate parameters or log
      print(f"OpenAI API request was invalid: {e}")
      pass
    except openai.error.AuthenticationError as e:
      #Handle authentication error, e.g. check credentials or log
      print(f"OpenAI API request was not authorized: {e}")
      pass
    except openai.error.PermissionError as e:
      #Handle permission error, e.g. check scope or log
      print(f"OpenAI API request was not permitted: {e}")
      pass
    except openai.error.RateLimitError as e:
      #Handle rate limit error, e.g. wait or log
      print(f"OpenAI API request exceeded rate limit: {e}")
      pass


    #print('OpenAI bundle received :\n', gpt_response)

    if gpt_response['choices'][0]['finish_reason'] == 'stop':
        gpt_text = gpt_response['choices'][0]['message']['content']

    else : 
        gpt_text = 'I had an issue processing your text.'

    tokens = gpt_response['usage']['total_tokens']
    cost_cents = round(0.2 / 1000 * tokens, 3)

    bundle = {
        'query_results' : query_results,
        'gpt_response' : gpt_text,
        'cost_cents' : cost_cents
    }

    return bundle