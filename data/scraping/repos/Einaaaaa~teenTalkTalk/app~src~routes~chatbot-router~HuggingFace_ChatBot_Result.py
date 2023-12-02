# -*- coding: utf-8 -*-
import requests
import pandas as pd
import numpy as np
import copy
import json
import torch
import pickle
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
import sklearn.manifold as manifold
import openai
import os
import sys
import csv
import json

from ast import literal_eval
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from transformers import pipeline
from transformers import GPT2TokenizerFast
from PIL import Image
from typing import List, Tuple, Dict
from dotenv import load_dotenv
''' 추가한 코드 : 챗봇 답변 시에 출력되는 경고문 삭제를 위한 코드 '''
import warnings # 경고문 없애기 위한 라이브러리
warnings.filterwarnings("ignore", message="Creating a tensor from a list of numpy.ndarrays is extremely slow.")


sys.stdout.reconfigure(encoding='utf-8')
''' 1. OpenAI API 불러오기 '''
load_dotenv()
openai.api_key = os.getenv("api_key")


''' 2. 데이터 불러오기 및 Embedding '''
# HuggingFace Embedding을 활용하여 Embdding vector 추출
data = pd.read_csv('./policy_data.csv', sep=",", dtype=str) # CSV파일 불러오기
data['recom_total'] = data['who'] + " / " + data['age'] + " / " + data['when'] + " / " + data['category'] # 정확한 추천을 위하여 who, age, when, category의 키워드 추출
model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2") # HuggingFace Embedding 모델
data['recom_embeddings'] = data['recom_total'].apply(lambda x : model.encode(x)) # 추천을 위한 Embedding
data['desc_embeddings'] = data['title'].apply(lambda x : model.encode(x)) # 설명을 위한 Embedding
data.to_csv('./data_embeddings.csv', encoding='utf-8-sig')



''' 3. Embedding된 데이터를 이용하여 가장 cosine값이 유사한 데이터 추출 '''
# top_k = 2 # 답변 개수 조절
def get_query_sim_top_k(query, model, df, top_k):
    query_encode = model.encode(query)
    cos_scores = util.pytorch_cos_sim(query_encode, df['recom_embeddings'])[0]
    top_results = torch.topk(cos_scores, k=top_k)
    return top_results

def get_overview_sim_top_k(desc, model, df, top_k):
    overview_encode = model.encode(desc)
    cos_scores = util.pytorch_cos_sim(overview_encode, df['desc_embeddings'])[0]
    top_results = torch.topk(cos_scores, k=top_k)
    return top_results

# query = "중학생을 위한 급식 관련 정책 추천해줘"
# top_result = get_query_sim_top_k(query, model, data, top_k)
# print(data.iloc[top_result[1].numpy(), :][['title', 'who', 'age', 'when']])



''' 4. OpenAI의 API를 이용하기 위하여 의도를 분류하기 위한 프롬포트 구성 '''
# 프롬프트 내용 수정
msg_prompt = {
    'recom' : {
                'system' : "너는 user에게 정책 추천을 도움주는 assistant입니다.",
                'user' : "당연하지!'로 시작하는 간단한 인사말 1문장을 작성해. 추천해주겠다는 말을 해줘.",
              },
    'desc' : {
                'system' : "너는 user에게 정책 설명을 도움주는 assistant입니다.",
                'user' : "'당연하지!'로 시작하는 간단한 인사말 1문장을 작성하여 user에게 정책을 설명해줘.",
              },
    'intent' : {
                'system' : "너는 user의 질문 의도를 이해하는 도움을 주는 assistant입니다.",
                'user' : "아래 문장은 'description','recommend', 중 속하는 categories만 보여라."
                }
}


# Intent 파악 함수
def set_prompt(intent, query, msg_prompt_init, model):
    '''prompt 형태를 만들어주는 함수'''
    m = dict()

    # 추천
    if ('recom' in intent):
        msg = msg_prompt_init['recom']

    # 설명
    elif 'desc' in intent:
        msg = msg_prompt_init['desc']

    # intent 파악
    else:
        msg = msg_prompt_init['intent']
        msg['user'] += f' {query} \n A:'

    for k, v in msg.items():
        m['role'], m['content'] = k, v
    return [m]

# OpenAI API 모델을 사용하기 위한 함수
def get_chatgpt_msg(msg):
    completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=msg
                    )
    return completion['choices'][0]['message']['content']

# 받은 Query를 이용하는 함수
def user_interact(query, model, msg_prompt_init):
    # 사용자의 의도를 파악
    user_intent = set_prompt('intent', query, msg_prompt_init, None)
    user_intent = get_chatgpt_msg(user_intent).lower()
    # print("user_intent : ", user_intent)

    # 사용자의 쿼리에 따라 prompt 생성
    intent_data = set_prompt(user_intent, query, msg_prompt_init, model)
    intent_data_msg = get_chatgpt_msg(intent_data).replace("\n", "").strip()\

    # 3-1. 추천이면
    if ('recom' in user_intent):
        recom_msg = str()

        # 유사 아이템 가져오기
        top_result = get_query_sim_top_k(query, model, data, top_k=1 if 'recom' in user_intent else 1) # 답변 개수 1개
        #print("top_result : ", top_result)

        # 검색이면, 자기 자신의 컨텐츠는 제외
        top_index = top_result[1].numpy() if 'recom' in user_intent else top_result[1].numpy()[1:]
        #print("top_index : ", top_index)

        # 정책명, 대상, 기간, 링크 데이터를 CSV 파일에서 불러오기
        r_set_d = data.iloc[top_index, :][['title', 'who', 'when','link']]
        r_set_d = json.loads(r_set_d.to_json(orient="records"))

        count = 0
        ''' 
        수정한 코드
        기존의 코드는 1개가 아닌 여러개의 리스트를 가져와 각각을 출력하는 코드로 이중 for문을 사용
        -> 우리는 1개 추천이기 때문에 1개의 for문을 사용하고 r_set_d의 index 0번의 item을 바로 가져오도록 수정
           (설명 파트도 동일하게 수정)
        '''
        recom_msg += "\n입력하신 내용 기반으로 가장 적합한 정책을 추천하겠습니다.\n"
        for _, v in r_set_d[0].items():
            if(count == 0):
                recom_msg += f"정책명 : '{v}'\n"
            elif(count == 1):
                recom_msg += f"대상 : {v}\n"
            elif(count == 2):
                recom_msg += f"기간 : {v}\n\n"
            elif(count == 3):
                recom_msg += "자세한 설명은 아래의 링크를 클릭하여 접속해보시기 바랍니다.\n"
                recom_msg += f"{v}\n"
            count += 1
                
        print(recom_msg)

    # 3-2. 설명이면
    elif 'desc' in user_intent:
        desc_msg = str()

        top_result = get_overview_sim_top_k(query, model, data, top_k=1)
        r_set_d = data.iloc[top_result[1].numpy(), :][['title','overview','link']]
        r_set_d = json.loads(r_set_d.to_json(orient="records"))

        count = 0
        desc_msg += "\n"
        for _, v in r_set_d[0].items():
            if(count == 0):
                desc_msg += f"{v} 정책이란 "
            elif(count == 1):
                desc_msg += f"{v} 하는 정책입니다.\n\n"
            elif(count == 2):
                desc_msg += "자세한 설명을 원하시면 아래의 링크를 클릭하여 접속해보시기 바랍니다.\n"
                desc_msg += f"{v}\n"
            count += 1

        print(desc_msg)
# query = input()
query = sys.argv[1]  # 첫 번째 커맨드 라인 인자로 전달된 쿼리
user_interact(query, model, copy.deepcopy(msg_prompt))
desired_answer = input("원하시는 답변이 되셨나요? (yes/no): ")
if desired_answer.lower() == "yes":
    print("더 궁금하신 것이 있다면 다시 질문해주세요.\n") # 출력 내용 수정
    exit(0)
else:
    query = input("정확한 답변을 원하시면 구체적인 키워드를 포함해서 질문해주세요: ")
    user_interact(query, model, copy.deepcopy(msg_prompt))