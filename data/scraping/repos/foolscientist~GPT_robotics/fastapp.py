import json
import time
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from fastapi.responses import *
from pydantic import BaseModel
from typing import Dict, List
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from starlette.requests import Request
from lang import DataBaseChat
import copy
import openai
import os
os.environ["OPENAI_API_KEY"] = ''
#自行更改代理端口，注意需使用美国节点
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
# openai.proxy = "http://127.0.0.1:7890"
dbchat=DataBaseChat()
app = FastAPI()

class Data(BaseModel):
    light:float=None
    wet:float=None
    temperature:float=None

class Query(BaseModel):
    qurey:str=None

#由于用户询问gpt
@app.post("/query")
def index(request: Request, query:Query):
    if(not query.qurey):
        return
    result=dbchat.chat(query.qurey)
    print(result)
    response=result['result']
    return {"response":response}
    

# #由于上报数据
# @app.post("/data")
# def index(request: Request, data:Data):
#     if (not(data.light and data.wet and data.temperature)):
#         print("1234567654321")
#         return
#     t=time.strftime('%m-%d %H:%M:%S')
#     with open('./data/realdata.txt', 'a') as f:
#         f.write(f"{{time:{t},light intensity:{data.light},humidity:{data.wet},temperature:{data.temperature}}}\n")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapp:app", host="127.0.0.1", port=8001, log_level="info")


