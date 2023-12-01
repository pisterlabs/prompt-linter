import os
import openai
import json
import numpy as np
from numpy.linalg import norm
import re
from time import time,sleep
from uuid import uuid4
import datetime




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


def timestamp_to_datetime(unix_time):
    return datetime.datetime.fromtimestamp(unix_time).strftime("%A,%B %d,%Y ag %I;%M%p %Z")


def gpt3_embedding(content,engine='text-embedding-ada-002'):
    content = content.encode(encoding='ASCII',errors='ignore').decode() #This is done because some unicode breaks gpt 3 thats why converting to ascii which is smaller code base and then converting back to unicode give output in correct format that is acceptable to gpt 3
    response= openai.Embedding.create(input=content,engine=engine)
    vector=response['data'][0]['embedding'] #this is a normal list
    return vector


def similarity(v1,v2): #gives us a better semantic similarity than just using dot product of two vectors
    #based upon https://stackoverflow.com/questions/18424228/cosine-similarity-2-number-lists
    return np.dot(v1,v2)/(norm(v1)*norm(v2)) #return cosine similarity


def fetch_memories(vector,logs,count): #take input as vectors of recent chats and count how many logs we have released in input
    scores=list()
    for i in logs:
        if vector==i['vector']:
            #skip this one because it is a messsage #skip the identical message
            continue
        score=similarity(i['vector'],vector)
        i['score']=score
        scores.append(i)
   
    ordered=sorted(scores,key=lambda d: d['score'],reverse=True) #we sort them to most relevant one
    # TODO- pick more memories temporally nearby the top most relevant memories
    try:
        ordered=ordered[0:count] #we order by n relevant memories
        return ordered
    except:
        return ordered
   
def load_convo(): #it loads the convo in the directory and storee it in a nexus and then sort it according to time
    files=os.listdir('nexus')
    files=[i for i in files if '.json' in i] #filter out any non json files
    result=list()
    for file in files:
        data=load_json('nexus/%s' % file)
        result.append(data)


    ordered = sorted(result,key=lambda d: d['time'],reverse=False) #sort them all chronologically
    return ordered


def summarize_memories(memories): #summarize a block of memories into one payload
    memories=sorted(memories, key= lambda d: d['time'],reverse=False) #sort them chronologically
    block=''
    identifiers=list()
    timestamps=list()
    for mem in memories:
        block += mem['memories'] +'\n\n' # basically the analogy is that during sleep our brain summarizes the memories of entire day in back ground but here instead of doing it in background we doing it in real time
        identifiers.append(mem['uuid'])
        timestamps.append(mem['time'])


    block=block.strip()
    prompt=open_file('prompt_notes.txt').replace('<<INPUT>>',block) #now this memories are then sent as prompt to open AI
    # TODO- do this in the background over time to handle huge amounts of memories
    notes= gpt3_completion(prompt)
    #### SAVE NOTES
    vector=gpt3_embedding(block)
    info={'notes':notes,'uuids':identifiers,'times':timestamps,'uuid':str(uuid4()),'vector':vector,'time':time()}
    filename='notes_%s.json' % time()
    save_json('internal_notes/%s' %filename, info)
    return notes
def get_last_messages(conversation,limit):
    try:
        short=conversation[-limit:] #gives last message
    except:
        short=conversation


    output=''
    for i in short:
        output+= '%s\n\n' %i['message']


    output=output.strip()
    return output


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
    openai.api_key=open_file('openaiapikey.txt')
    while True:
        #### get user input,save it,vectorize it , etc.
        input_by_user=input('\n\nUSER:')
        timestamp=time()
        vector=gpt3_embedding(input_by_user)
        timestring=timestamp_to_datetime(timestamp)
        message='%s:%s-%s' %('USER',timestring,input_by_user)
        info={'speaker':'USER','time':time(),'vector':vector,'message':message,'uuid':str(uuid4())}
        filename='log_%s_USER.json'% time()
        save_json('chat_logs/%s' %filename,info)
        ####load conversation
        conversation = load_convo()
        #### compose corpus(fetch_memories,etc)  #corpus is basically all the memories and context needed to do cognitive labour that means without any exteernal help
        memories=fetch_memories(vector,conversation,10) #pull episodic memories
        # TODO- fetch declarative memories (facts,wikis,KB,internet etc)
        notes=summarize_memories(memories)
        recent=get_last_messages(conversation,4) #getting 4 recent chat back and forth to get the context of the conversation
        prompt=open_file('prompt_response.txt').replace('<<NOTES>>',notes).replace('<<CONVERSATION>>',recent)
        #### generate response,vectorize,save etc
        output=gpt3_completion(prompt)
        timestamp=time()
        timestring=timestamp_to_datetime(timestamp)
        message='%s:%s-%s' %('AIDA',timestring,output)
        vector=gpt3_embedding(output)
        info={'spaker':'aida','time':time(),'vector':vector,'message':output,'uuid':str(uuid4())} # a dictionary object is created
        filename='log_%s_aida.json' % time()
        save_json('chat_logs/%s' % filename,info) # %s is used to define that the file is stored in the same format
        #printing output
        print('\n\nAIDA:%s' %output)


## THIS IS SIMPLEST IMPLEMENATION OF NATURAL LANGUAGE COGNITIVE ARCHITECTURE
















