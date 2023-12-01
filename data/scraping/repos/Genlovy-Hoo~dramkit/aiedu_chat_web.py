# -*- coding: utf-8 -*-

# 启动方式：
# 1. python main.py
# 2. uvicorn main:app --reload

import os
import time
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates

from dramkit.gentools import isnull, tmprint
from dramkit.iotools import make_dir
from dramkit.openai.aiedu_chat import OpenAIChat

app = FastAPI()

WAIT_SECONDS = 60*10
global CONVERS
CONVERS = {} # {ip: {'last_tm': tm, 'chater': OpenAIChat()}}

# CONVERS_DIR = None
CONVERS_DIR = './conversations/'
if not isnull(CONVERS_DIR):
    make_dir(CONVERS_DIR)


def clean_convers(CONVERS):
    res = {}
    for ip in CONVERS:
        if time.time()-CONVERS[ip]['last_tm'] <= WAIT_SECONDS:
            res[ip] = CONVERS[ip]
    return res


templates = Jinja2Templates(directory='templates')


# 添加首页
@app.get('/')
async def home(req: Request):
    return templates.TemplateResponse(
            'index.html',
            context={'request': req,
                     'ans': '',
                     'err': ''})


# 网页提交聊天
@app.post('/webchat')
async def webchat(req: Request, prompt: str = Form(None)):
    print('question:')
    tmprint(prompt)
    ip = req.client.host
    tmnow = time.time()
    global CONVERS
    if ip not in CONVERS or tmnow-CONVERS[ip]['last_tm'] > WAIT_SECONDS:
        CONVERS[ip] = {'last_tm': tmnow, 'chater': OpenAIChat()}
    CONVERS[ip]['last_tm'] = time.time()
    CONVERS = clean_convers(CONVERS)
    print('The remaining IPs with conversation:', CONVERS.keys())
    save_path = None
    if not isnull(CONVERS_DIR):
        save_path = os.path.join(CONVERS_DIR, '%s.json'%ip)
    result, error_info = CONVERS[ip]['chater'].chat(
                         prompt, save_path=save_path)
    print('answer:')
    print(result)
    print('error info:')
    tmprint(error_info)
    return templates.TemplateResponse(
            'index.html',
            context={'request': req,
                     'ans': result,
                     'err': error_info})


if __name__ == '__main__':
    uvicorn.run(app, host='10.0.0.4', port=8080)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
