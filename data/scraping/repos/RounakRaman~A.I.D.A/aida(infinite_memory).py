'''This version of AIDA is supported by vector database called Pinecone it will further enhance the ability 
of this voice enabled assistant by increasing the use of semantic search of the vectors.So Stay Tuned'''

#we wont be needing notes in this cause we will be making it having super-long memory

# Also pinecone allows us to save some metadeta

import os
import openai
import json
import numpy as np
from numpy.linalg import norm
import re
from time import time,sleep
from uuid import uuid4
import datetime
import pinecone


def open_file(filepath):
    with open(filepath,'r',encoding='utf-8') as infile:
        return infile.read()
   
def save_file(filepath,content):
    with open(filepath,'w',encoding='utf-8') as outfile:
        outfile.write(content)


def load_json(filepath):
    with open(filepath,'r',encoding='utf-8') as infile:
        return json.load(infile)
   
def save_json(filepath):
    with open(filepath,'w',encoding='utf-8') as outfile:
        json.dump(payload,outfile,ensure_ascii=False,sort_keys=True,indent=2)

def load_conversation(results): #this function return the most relevant convo we had with aida
    result=[]
    for m in results['matches']:
        info=load_json('nexus%s.json' %m['id']) #this will load the metadata
        result.append(info)
    
    ordered=sorted(result,key=lambda d: d['time'], reverse=False) #sort them chronologically
    messages=[i[message] for i in ordered ] #this is list comprehension that is making a smaller list instead of a nested list
    messageblock='\n'.join(messages).strip()
    return messageblock



def timestamp_to_datetime(unix_time):

    return datetime.datetime.fromtimestamp(unix_time).strftime("%A,%B %d,%Y ag %I;%M%p %Z")

def gpt3_embedding(content,engine='text-embedding-ada-002'):
    content = content.encode(encoding='ASCII',errors='ignore').decode() #This is done because some unicode breaks gpt 3 thats why converting to ascii which is smaller code base and then converting back to unicode give output in correct format that is acceptable to gpt 3
    response= openai.Embedding.create(input=content,engine=engine)
    vector=response['data'][0]['embedding'] #this is a normal list
    return vector

def gpt3_completion(prompt,engine='text-davinci-003',temp=0.0,top_p=1.0,token=400,freq_pen=0.0,pres_pen=0.0,stop=['USER:','AIDA:']): # here stops are AIDA and user otherwise it will keep on talking
    max_retry=5
    retry=0
    prompt=prompt.encode(encoding='ASCII',errors='ignore').decode()
    while True:
        try:
            response=openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=token,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text=response['choice'][0]['text'].strip()
            text=re.sub('[\r\n]+','\n',text)
            text=re.sub('[\t]+','',text)
            filename='%s_gpt3.txt' %time()
            if not os.path.exists('gpt3_logs'):
                os.makedirs('gpt_logs')
        except Exception as oops:
            retry +=1
            if retry>=max_retry:
                return "GPT 3 error:%s" %oops
           
            print('Error communicating with OpenAI:',oops)
            sleep(1)

if __name__=='__main__':
    convo_len=15
    openai.api_key=open_file('openaiapikey.txt') #authorising the key with open ai to access large language models
    pinecone.init(api_key=open_file('pinecone_key.txt'), environment="us-west1-gcp-free") #authorizing the key with pinecone to access the vector storage feature on a cloud basd region
    vector_database= pinecone.Index("aida-mvp") #it is also called as index
    while True:
        payload=list() #create a list container that will later store meeesage and vector as tupe to then be upserted to pinecone
        #### get user input,save it,vectorize it , save it to pinecone etc.
        input_by_user=input('\n\nUSER:')
        timestamp=time()
        timestring=timestamp_to_datetime(timestamp)
        message='%s:%s-%s' %('USER',timestring,input_by_user)
        vector=gpt3_embedding(message)
        uniqueid=str(uuid4())
        metadata={'speaker':'USER','time':time(),'message':message,'uuid':uniqueid}
        save_json('chat_logs/%s' %uniqueid,metadata) #we are saving the unique ids locally so that pinecone can search the file and tell us which uuids to grab later.Now since its is saved locally we dont need values or metadata
        payload.append((uniqueid,vector,metadata))
        results= vector_database.query(vector=vector,
                                   top_k=convo_len, #top_k determines how many similar matches you want to find and in this case it is 15
                                   include_values=False, # does not return the vector values
                                   include_metadata=False
                                   # does not return the metadata that consists of messages
                                 ) 
        conversation=load_conversation(results) #result should be a dict with matches which is a list of dicts with 'id'
        prompt=open_file('prompt_response.txt').replace('<<CONVERSATION>>',conversation)
        #thus this will not makes notes but we will take the conversation infinetly far but pull only recent 15 messages as mentioned in top_k
        output=gpt3_completion(prompt)
        timestamp=time()
        timestring=timestamp_to_datetime(timestamp)
        message='%s:%s-%s' %('AIDA',timestring,output)
        vector=gpt3_embedding(message)
        uniqueid=str(uuid4())
        metadata={'spaker':'aida','time':time(),'vector':vector,'message':output,'uuid':uniqueid} # a dictionary object is created
        save_json('nexus/%s.json' %uniqueid,metadata)
        payload.append((uniqueid,vector))
        vector_database.upsert(payload)
        #print output
        print('\n\nAIDA: %s' %output)


    #storing the vectors at the end of both user and aida so that we accidentally dont pull our most recent messsages 
    #To ingest vectors into your index, we use the upsert operation.
    #The upsert operation inserts a new vector in the index or updates the vector if a vector with the same ID is already present.
    # Also take care that when upserting larger amounts of data, upsert data in batches of 100 vectors or fewer over multiple upsert requests.
    #now search for relevant messages,and generate a response
    
    
