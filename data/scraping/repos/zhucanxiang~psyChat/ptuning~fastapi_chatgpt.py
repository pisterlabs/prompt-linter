from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from typing import Optional
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import time
import copy

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static") # 挂载静态文件，指定目录
templates = Jinja2Templates(directory="templates") # 模板目录

history_dir = "history/"

open_api_key = "your_openai_key"


def creat_conversation():
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "你是一名专业的心理咨询师，在给人类患者做心理咨询，请循序渐进地了解人类患者的心理情况，请以聊天的方式回复，每次回复不要超过60字。你就是专业的心理咨询师，不要建议用户找人类咨询师寻求帮助"
        ),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    print(prompt)
    llm = ChatOpenAI(temperature=0, openai_api_key=open_api_key)
    memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)
    return conversation

conversation = creat_conversation()

def write_chat_history(username, patient_say, doctor_say):
    filename = history_dir + username + '.txt'
    with open(filename, "a+", encoding='utf-8') as f:
        patient_say = '病人:' + patient_say
        doctor_say = '医生:' + doctor_say
        f.write(patient_say + '\n')
        f.write(doctor_say + '\n')

    chat_history_str = ""
    with open(filename, "r", encoding="utf-8") as f:
        for line in f.readlines():
            chat_history_str += line
    return chat_history_str

def read_chat_history(username):
    filename = history_dir + username + '.txt'
    chat_histories = []
    with open(filename, "r", encoding='utf-8') as f:
        chat_item = {}
        for line in f.readlines():
            if line.startswith('医生:'):
                chat_item['doctor'] = line.replace('医生:', '')
                new_chat_item = copy.deepcopy(chat_item)
                print(chat_item['doctor'])
                chat_histories.append(new_chat_item)
                chat_item = {}
            else:
                chat_item['patient'] = line.replace('病人:', '')
        if 'doctor' in chat_item or 'patient' in chat_item:
            chat_histories.append(chat_item)
    return chat_histories

def load_chat_history(username):
    filename = history_dir + username + '.txt'
    is_new_user = False
    chat_history_str = ""
    if os.path.exists(filename):
        #老用户，则读取聊天记录
        is_new_user = False
        with open(filename, "r", encoding='utf-8') as fr:
            for line in fr.readlines():
                chat_history_str += line
    else:
        # 新用户，则创建聊天记录文件
        is_new_user = True
        with open(filename, "w", encoding='utf-8') as fw:
            doctor_say = '医生:你好, {}, 我是王医生'.format(username) + '\n'
            fw.write(doctor_say)
            chat_history_str =  doctor_say
            fw.flush()
            fw.close()
    return is_new_user, chat_history_str


class PredictData(BaseModel):
    query: str       # 输入
    username: str    # username
    #max_length: Optional[int] = 4096 #最大长度
    #top_p:  Optional[float] = 0.7
    #temperature: Optional[float] = 0.95

class ClearHistoryData(BaseModel):
    username: str    # username

@app.get("/home/{username}")
def home(request:Request, username:str):
    #new_user, chat_history_str = load_chat_history(username)
    chat_history_str = ''
    return templates.TemplateResponse("index_h.html", {"request": request, "username": username, "chat_history": chat_history_str})


def predict(query):
    return conversation.predict(input=query)

@app.post("/predict")
def chat(request: PredictData):
    response = predict(request.query)
    chat_history_str = write_chat_history(request.username, request.query, response)
    return_data = {'response': response, "chat_history": chat_history_str}
    return return_data


@app.post("/clear_history")
def clear_history(request:ClearHistoryData):
    username = request.username
    response = "成功清空{}的历史聊天记录".format(username)
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8082)



