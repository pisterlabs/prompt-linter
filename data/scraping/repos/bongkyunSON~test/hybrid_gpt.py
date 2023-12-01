from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import hashlib
import hmac
import base64
import time
import requests
import json
import openai
import os
from dotenv import load_dotenv



class gptMessangeSender:
    # load environment
    load_dotenv()

    # gpt keys
    openai.organization = os.getenv("OPENAI_ORG_KEY")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    def GPT(msg):
        response = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
            messages=[{"role":"system", "content":"너는 나의 모든 분야의 선생님이야 친절한 말투로 대답해주고 200자 이내로 대답해줘"}, {"role":"user", "content":msg}],
            max_tokens=500,
            temperature=0.5
        )
        response = response.to_dict_recursive()
        response = response["choices"][0]["message"]["content"]
        return response


class ChatbotMessageSender:
    # load environment
    load_dotenv()
    # chatbot api gateway url & secret key
    ep_path = os.getenv("CLOVA_PATH")
    secret_key = os.getenv("CLOVA_SECRET_KEY")

    def req_message_send(self, msg):

        timestamp = self.get_timestamp()
        request_body = {
            'version': 'v2',
            'userId': 'tester',
            'timestamp': timestamp,
            'bubbles': [
                {
                    'type': 'text',
                    'data': {
                        'description': msg
                    }
                }
            ],
            'event': 'send'
        }

        ## Request body
        encode_request_body = json.dumps(request_body).encode('UTF-8')

        ## make signature
        signature = self.make_signature(self.secret_key, encode_request_body)

        ## headers_clova
        custom_headers = {
            'Content-Type': 'application/json;UTF-8',
            'X-NCP-CHATBOT_SIGNATURE': signature
        }

        ## POST Request
        response = requests.post(headers=custom_headers, url=self.ep_path, data=encode_request_body)
        return response

    @staticmethod
    def get_timestamp():
        timestamp = int(time.time() * 1000)
        return timestamp

    @staticmethod
    def make_signature(secret_key, request_body):

        secret_key_bytes = bytes(secret_key, 'UTF-8')

        signing_key = base64.b64encode(hmac.new(secret_key_bytes, request_body, digestmod=hashlib.sha256).digest())

        return signing_key



class Input_msg(BaseModel):
    msg : str


app = FastAPI()
@app.post("/chatbot")
async def hybrid_chatbot(input_text: Input_msg):

    res = ChatbotMessageSender().req_message_send(msg=input_text.msg)
    if(res.text.find('"value":"canNotHelpMsg"') > 0 or res.text.find('"value":"similarAnswer"') > 0 or res.text.find('"value":"unknownMsg"') > 0):
         input_text = input_text.dict()
         gpt_answer = gptMessangeSender.GPT(msg=input_text["msg"])
         gpt_answer = gpt_answer.replace("/n:"," ")
         return {"msg": gpt_answer}
    
    elif (res.text.find('"imageUrl":') > 0):
        ### image 답변일 경우 ###
        reponse = json.loads(res.text)
        if "imageUrl" in reponse:
            image_url = reponse["imageUrl"]
            print("imageUrl:", image_url)

        # "description" 키의 값을 저장할 리스트
        korean_descriptions = [] 

        # JSON 데이터에서 "description" 키의 값 추출
        def extract_description(data):
            if isinstance(data, dict):
                for key, value in data.items():
                    if key == "description" and isinstance(value, str):
                        korean_descriptions.append(value)
                    else:
                        extract_description(value)
            elif isinstance(data, list):
                for item in data:
                    extract_description(item)

        # JSON 데이터에서 "description" 키의 값 추출
        extract_description(reponse)
        for description in korean_descriptions:
            description
        #return ('\n'.join(korean_descriptions))
        korean_descriptions = '\n'.join(korean_descriptions)
        return JSONResponse({"msg": korean_descriptions})


        ### 답변에 이미지가 없을 경우 ###
    else:
        reponse = json.loads(res.text)
        
        # "description" 키의 값을 저장할 리스트
        korean_descriptions = []

        # 재귀적으로 JSON 데이터를 탐색하여 "description" 키의 값 추출
        def extract_description(data):
            if isinstance(data, dict):
                for key, value in data.items():
                    if key == "description" and isinstance(value, str):
                        korean_descriptions.append(value)
                    else:
                        extract_description(value)
            elif isinstance(data, list):
                for item in data:
                    extract_description(item)

        # JSON 데이터에서 "description" 키의 값 추출
        extract_description(reponse)

        # 추출된 "description" 값 출력
        for description in korean_descriptions:
            korean_descriptions

        #return ('\n'.join(korean_descriptions))
        korean_descriptions = '\n'.join(korean_descriptions)
        return JSONResponse({"msg": korean_descriptions})

            
