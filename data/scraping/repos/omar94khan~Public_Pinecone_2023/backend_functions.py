import json
import openai
import tiktoken
import pinecone
import cohere
import ast
import numpy as np
import pandas as pd
from datetime import datetime
import re
import utils

cohere_api_key = utils.cohere_api_key
pinecone_api_key = utils.pinecone_api_key
pinecone_environment = utils.pinecone_environment
openai.api_key  = utils.openai_api_key

# initialize pinecone and connetct to the index
pinecone_environment = "us-west4-gcp-free"
pinecone.init(pinecone_api_key, environment=pinecone_environment)
pinecone_index_name = "salman-khan"
index = pinecone.Index(pinecone_index_name)

# initializing cohere client
co = cohere.Client(cohere_api_key)

# Prompts
delimiter = "####"
current_time = datetime.now()

category_prompt = f"""You will be provided with a query. \
The user query will be delimited with {delimiter} characters.
your task is to classify each query into a category, either time related or general log query. \
Time related queries are those where there is a mention of a time or time period like month, year, day or any date etc. \

Provide your output in json format where \
keys: category.
values: T for Time related query and G for general log query

Examples for time related queries: Where were you last night, how much you spent on grocery this month?, what was the last entry in journal
Examples for general log queries: What I like to have in dinner?, where have I spent most money, what book I read the most"""

time_prompt = f"""You will be provided with a query. \
The user query will be delimited with {delimiter} characters.
The user will provide a time-bound query with some timeframe in it. for example, today, yesterday, this week, 6 days ago, three months ago, last year, between january and march,first quarter etc. \
Detect the timeframe from the query and produce start and end date.  \
Today's date is {current_time}. Use today's date as an end date if no end date is detected in the query. \
Otherwise use end date specified in a user's query. \
The start date is based on user's query. Don't share code just give the output in json format. \

In json the values which are the dates should not contain any '-' and should be in format YYYYMMDD.\

Provide your output in json format for Example: ('start_date' : 20210101, 'end_date': 20220108)"""

answering_prompt = f"""you are a personal assistant for a user. The user has recorded journal entries along with timestamps which may contain\
information about the user's experiences, thoughts, activities, personal reflections, descriptions of events, financial transactions, or any \
relevant details. Each unique journal entry will be divided by a delimitter {delimiter}.\

Your task is to answer a user query using those journal entries as context. You should provide insightful and accurate responses based on the\
information available in the journal entries.

The user wants to retrieve information about a particular event mentioned in their journal entries. The user query is classified into two categories,\
Time related or General Log query. If the category is T, it means the query\
is time related and if it is G, it means it is a general log query.\

For general and time related log queries, provide to the point but accurate response based on the information provided in the context. \
If there is no keyword matching in the query"""

log_query_prompt = f"""you are a personal assistant for a user. The user is either going to record journal entries along with timestamps which may contain\
information about the user's experiences, thoughts, activities, personal reflections, descriptions of events, financial transactions, or any \
relevant details. Or the user is going to ask a query regarding user's existing records. \
Each unique journal entry will be divided by a delimitter {delimiter}.\

Your task is to classify whether the given command is a query or a journal entry. \
Mostly user queries are interrogative, or commanding in nature as compared to journal entries. \


Examples for general journal entries: I did not go to pakistan tour, I had a meeting from 2-3pm, I cried yesterday. \
Examples for user queries: What I like to have in dinner?, where have I spent most money, what book I read the most \

Provide your output in string values: 'Q' for query and 'E' for general log entry"""



# function to generate response from GPT
def get_completion_from_messages(messages, model="gpt-3.5-turbo-16k", temperature=0, max_tokens=500):
    response = openai.ChatCompletion.create(model=model, messages=messages,
                                            temperature=temperature, max_tokens=max_tokens)
    return response.choices[0].message["content"]

# find similar general entries
def find_similar(query, namespace='test'):
  xq = co.embed(texts=[query], model='large', truncate='LEFT').embeddings
  similar = index.query(xq, top_k=300, include_metadata=True, namespace=namespace)
  texts = []
  for i in similar['matches']:
    texts.append(i.metadata['text'])
  return texts


# find similar time related entries
def find_similar_time(query, startDate, endDate, namespace='test'):
  conditions = {'date': {'$gte': startDate, '$lte': endDate}}
  xq = co.embed(texts=[query], model='large', truncate='LEFT').embeddings
  similar = index.query(queries=xq, top_k=30, filter=conditions, include_metadata=True, namespace=namespace)
  texts = [match['metadata']['text'] for match in similar['results'][0]['matches']]
  return texts

# answer general query
def genQuery(query, category, namespace='test'):
  current_time = datetime.now()
  query = f'{current_time} {query}'
  context = find_similar(query, namespace=namespace)
  context_ = " #### ".join(context)

  messages =  [{'role':'system', 'content': category},
               {'role':'system', 'content': answering_prompt},
               {'role':'user', 'content': context_},
               {'role':'user', 'content': query}]
  response = get_completion_from_messages(messages)
  return response

# answer time related query
def timeQuery(query, category, startDate, endDate, namespace='test'):
  current_time = datetime.now()
  query = f'{current_time} {query}'
  context = find_similar_time(query,startDate,endDate, namespace=namespace)
  context_ = " #### ".join(context)

  messages =  [{'role':'system', 'content': category},
               {'role':'system', 'content': answering_prompt},
               {'role':'user', 'content': context_},
               {'role':'user', 'content': query}]
  response = get_completion_from_messages(messages)
  return response

# function for any type of query
def ask_PA(query, namespace='test'):
  msg_ctg =  [{'role':'system', 'content': category_prompt},
              {'role':'user', 'content': query}]
  category_value = get_completion_from_messages(msg_ctg)
  category = json.loads(category_value)['category']
  
  if category == "T":
    time_ctg =  [{'role':'system', 'content': time_prompt},
              {'role':'user', 'content': query}]
    timeframe = get_completion_from_messages(time_ctg)

    startDate = json.loads(timeframe)['start_date']
    startDate = float(startDate)
    endDate = json.loads(timeframe)['end_date']
    endDate = float(endDate)
    return timeQuery(query, category, startDate, endDate, namespace=namespace)
  else:
    return genQuery(query, category, namespace=namespace)

# function to add new journal entries
def new_entry(entry, namespace='test'):
  current_time = datetime.now().strftime('%Y/%m/%d %H:%M')
  entry = f'Time: {current_time}, Entry: {entry}'
  emb = co.embed(model='embed-english-v2.0', texts=[entry]).embeddings
  emb = [[float(e) for e in sublist] for sublist in emb]
  index_stats = index.describe_index_stats()
  try:
    vector_count = index_stats['namespaces'][namespace]['vector_count']
    ids = str(vector_count)
  except KeyError:
    ids = '0'
  date = re.search(r'Time: (\d{4}/\d{2}/\d{2})', entry).group(1)
  date = float(date.replace('/', ''))
  meta = {'date': date, 'text': entry}
  to_upsert=[{'id': ids, "values":emb[0], "metadata": meta}]
  index.upsert(vectors=to_upsert,namespace=namespace)
  return "Entry added successfully"

# final PA
def final_PA(query, namespace = 'test'):
    messages =  [{'role':'system', 'content': log_query_prompt},
               {'role':'user', 'content': query}]
    response = get_completion_from_messages(messages)

    if response == 'Q':
      return ask_PA(query, namespace = namespace)

    elif response == 'E':
      return new_entry(query, namespace= namespace)

    else:
      return str('Please enter your input')

def access_entries(namespace,k=25):
    entries = index.query(namespace=namespace,top_k=10000,id='0',include_metadata=True)
    sorted_data = sorted(entries['matches'], key=lambda x: int(x['id']))
    result = [entry['metadata']['text'] for entry in sorted_data[-k:]]


    # Initialize empty lists to store 'Time' and 'Entry' values
    time_list = []
    entry_list = []

    # Extract 'Time' and 'Entry' values from each string and append them to the respective lists
    for entry in result:
        parts = entry.split(', Entry: ', maxsplit = 1)
        time = parts[0].replace('Time: ', '')
        entry_text = parts[1]
        time_list.append(time)
        entry_list.append(entry_text)

    # Create a dataframe using the lists
    df = pd.DataFrame({'Time': time_list, 'Entry': entry_list}).set_index('Time').sort_index(ascending=False)
    return df