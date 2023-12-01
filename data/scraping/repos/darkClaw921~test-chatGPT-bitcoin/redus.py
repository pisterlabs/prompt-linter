import redis
import langchain.schema
from langchain.memory import RedisChatMessageHistory
from dotenv import load_dotenv
from pprint import pprint
import os
from langchain.memory import ConversationSummaryMemory, ChatMessageHistory
from langchain.llms import OpenAI
from langchain.chains import LLMChain
load_dotenv()
from langchain.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)

from langchain.chat_models import ChatOpenAI

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import json
import openai
from langchain.memory import RedisChatMessageHistory
from langchain.schema import messages_from_dict, messages_to_dict
from loguru import logger
from langchain.memory.chat_message_histories import RedisChatMessageHistory

@logger.catch
def prepare_history(lst:list, promtText:str):
    #promt = SystemMessage(content="ты менеджер продажник, отечай так чтобы клиент точно что-то купил"), 
    promt = SystemMessage(content=promtText)
    history = [[promt]]
    print('lst')
    pprint(lst)
    for index,item in enumerate(lst):
        human = None
        ai = None
        form= type(item)
        if form == langchain.schema.HumanMessage:
            human = item.content     
            print('человек ', human)
        elif form == langchain.schema.AIMessage:
            ai = item.content
            print('ai ',ai)
        
        if len(history[0]) == 1 and ai == None:
            history[0].append(HumanMessage(content=human)) 

        elif form == langchain.schema.HumanMessage:
            try:
                history[index].append(AIMessage(content=human))
            except:
                history.append([AIMessage(content=human)])

        elif form == langchain.schema.AIMessage:
            try:
                history[index].append(AIMessage(content=ai))
            except:
                history.append([AIMessage(content=ai)])
                
        else:
            history[index].append([HumanMessage(content=human), AIMessage(content=ai)])


    print(f'{history=}')
    return history 
        #pprint(history)
#history=[[(SystemMessage(content='ты менеджер продажник, отечай так чтобы клиент точно что-то купил', additional_kwargs={}),)],
#          [HumanMessage(content='hi', additional_kwargs={}, example=False)]]

#history = RedisChatMessageHistory("14561")

r = redis.Redis(host='localhost', port=6379, decode_responses=False)


#r1 = {"role": "system", "content": 'ты менеджер продажник, отечай так чтобы клиент точно что-то купил'}
r2 = {"role": "user", "content": 'что за акция?'}

#r.lpush('1234', json.dumps(r1))
r.lpush('1234', json.dumps(r2))

#items = r.lrange('1234', 0, -1)
def add_message_to_history(userID:str, message:str):
    r.lpush(userID, json.dumps(message))
#history = [json.loads(m.decode("utf-8")) for m in items[::-1]]
#print(history)
def get_history(userID:str):
    items = r.lrange(userID, 0, -1)
    history = [json.loads(m.decode("utf-8")) for m in items[::-1]]
    return history
history = get_history('1234')

openai.api_key = key
completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=history,
    temperature=1,
    )

answer = completion.choices[0].message.content
print(answer)
r.lpush('1234', json.dumps({"role": "assistant", "content": answer}))
#r.hmset('1234', r1)
#r.hmset('1234', r2)
#a = r.hmset('1234', conversation)
#answer = conversation.predict(input="меня зовут Игорь")
#print(answer)
#exit()
#print(f"business: {r.hgetall('1234')}")


#a = chat.generate(batch_messages)
#print(memory)
#a = conversation.run("good")
#print(a)
#print(history.clear())
#r = redis.Redis(host='localhost', port=6379, decode_responses=True)
#r.set('foo1', 'bar')
# True
#a = r.get('foo')
# bar
#print(a)