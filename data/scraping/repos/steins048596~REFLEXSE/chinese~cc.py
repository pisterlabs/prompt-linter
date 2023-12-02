# -*- coding:utf-8 -*-
import paddle
import os 
#os.environ['http_proxy']="http://127.0.0.1:7892"
#os.environ['https_proxy']="http://127.0.0.1:7892"
paddle.device.set_device('gpu:3')
import openai
openai.api_key = "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"
import json
import csv
from collections import Counter
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from heapq import nlargest
from paddlenlp.datasets import load_dataset
import simplejson as json

import os
import paddlenlp
import numpy as np
import paddle.optimizer as opt
from paddlenlp.transformers import ErnieModel, ErnieTokenizer
import paddle.nn.functional as F
from paddle import nn
from paddlenlp.transformers import ErniePretrainedModel
from paddlenlp.transformers import AutoModel, AutoTokenizer
class ValueErnie(ErniePretrainedModel):
    def __init__(self, config,dropout=None):
        super(ValueErnie, self).__init__(config)        
        self.ernie = ErnieModel(config)
        self.value_head = nn.Linear(self.ernie.config['hidden_size'], 1)
        self.dropout=nn.Dropout(dropout if dropout is not None else
                                  self.ernie.config["hidden_dropout_prob"])

    def forward(self, chosen_ids, chosen_mask):
        chosen_ids=chosen_ids.squeeze(1)
        chosen = self.ernie(chosen_ids,attention_mask=chosen_mask)
        chosen_output = self.dropout(chosen[0])
        chosen_logits = self.value_head(chosen_output)
        chosen_reward=chosen_logits.mean(axis=1).squeeze(1)
        return chosen_reward
       
MODEL_NAME2='ernie-health-chinese'
tokenizer = paddlenlp.transformers.AutoTokenizer.from_pretrained(MODEL_NAME2)
model = ValueErnie.from_pretrained('little_model/real_model/ernie-health-chinese/',dropout=0.1)
load_layer_state_dict = paddle.load("little_model/real_model/zhusu.pdparams")
model.set_state_dict(load_layer_state_dict)


def modelinput(text):
    model.eval()
    input_ids = tokenizer(text=text, max_seq_len=512)['input_ids']
    input_ids=paddle.to_tensor(input_ids, dtype='int64')
    input_ids=paddle.unsqueeze(input_ids,axis=0)
    input_mask = np.ones(len(input_ids[0]),dtype='int64')
    input_mask=paddle.to_tensor(input_mask, dtype='int64')
    results = model(input_ids,input_mask).item()
    return results

def topn_dict(d, n):
    return nlargest(n, d, key=lambda k: d[k])

def save_csv(total,i,rname):
    with open(rname, 'a', newline='') as file:
        writer = csv.writer(file)
        if i==0:
            writer.writerow(["id", "text",])
        writer.writerow(["dev_"+str(i),total])
rname='new_experiment/cc_chatgpt_fewshot.csv'

print("a")

data_output=[]
present_total=[]
text_total=[]
filename='data/test_index.csv'
with open(filename,encoding='gbk') as f:
        i=0    
        f_csv=csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            present_total.append(str(row[7]))
            text_total.append(str(row[4]))
            i=i+1
temp=[]
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
def completion_with_backoff(prompt1,prompt2,prompt3,icl1,icl2,temperature):
    result=openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=[
                    {"role": "system", "content": "你是一位专业医生，请根据具体的规则和要求修改电子病历"},
                    {"role": "user", "content": prompt1+icl1+icl2+prompt2+prompt3}
                ],
                temperature=temperature
            )
    return result
i=0
xr=0
#i=316
temptext=[]
black_list=[]
lack_list=[15,149]
temptext.append([])
dict_temp={}
count0=0
flag=0
flag_text=0
flag_text0=0
#while(i<=1):
while(i<=len(text_total)):
    print("I:",i)
# 设置 API 密钥
    if flag_text==0:
        text00=str(text_total[i]).replace('(1)主诉：','')
    temperature=0.2
    print(text00)
    present=str(present_total[i]).replace('(2)现病史：','')
    if flag_text==0:
        temptext[-1].append(text00.replace('。',''))
        dict_temp.setdefault(text00.replace('。',''))
        dict_temp[text00.replace('。','')]=modelinput(present+'\n'+str(text_total[i]))
#    print(present)
    icl1="(1)主诉：婴儿咳嗽持续4-5天。\nErrorPrompt：\n禁止模糊天数，精确记录持续时间\nreference：\n咳嗽5天。\n"
    icl2="(1)主诉：8个半月的宝贝打完疫苗第二天就开始拉肚子了，已经持续一周了。\nErrorPrompt：\n只保留前2个主要症状，控制在15字以内。\nreference：\n腹泻一周。\n"
    prompt1="规则：主诉是指促使患者就诊的 主要症状 ( 或体征 ) +部位+性质+持续时间（“提示”中没有时间天数则“主诉”不写持续时间）。（简明扼要，一般不超过15字）症状多于一项时，需按时间先后顺序依次列出，一般只保留前2个主要症状，只保留前2个主要症状。禁止用数天、数月等模糊不清的词，要具体到几天、几月、几年（也不可写3.5年等类似词表示，可以改为月份表达）。\nErrorPrompt尽量简短,reference控制在15字以内。\n"
#    if len(str(text_total[i]).replace('(1)主诉：',''))>10:
    prompt2="“提示”："+str(text_total[i]).replace('(1)主诉：','')+'\n'
#    else:
#        prompt2="“提示”："+present+'\n'
    prompt3="参考规则，根据下方“主诉”生成ErrorPrompt与reference，并对比“提示”和下方“主诉”，“主诉”中禁止生成“提示”以外的信息，“reference”中禁止生成“提示”以外的信息\n"+"(1)主诉："+text00+"\nErrorPrompt"   
    # 调用 ChatGPT
#    if len(temptext[-1])<2:
#        print(present)
    if flag==0:
        response = completion_with_backoff(prompt1,prompt2,prompt3,icl1,icl2,temperature)
    if flag==1:
        response = completion_with_backoff(prompt1+'\n',prompt2,prompt3,icl2,icl1,temperature)
    text0=str(response["choices"][0]["message"]["content"])
    print(text0)
    flag_text=1 
    if "reference" not in text0 and "改为" not in text0:
        if "无错误" not in text0 and "无需修改" not in text0 and "无误" not in text0 and "不需要" not in text0:
            flag=flag^1
            continue
        else:
            flag_text0=2       
    if "reference:" in text0 or "reference：" in text0:
        if "reference:" in text0:
            text=text0[text0.find('reference:'):]
        if "reference：" in text0:
            text=text0[text0.find('reference：'):]
        if "症状" in text or "描述" in text:
            print("b")
            text00=max(dict_temp, key=lambda x: dict_temp[x])
            flag=flag^1
            continue
        if '。' in text:
            text=text[:text.find('。')].replace('reference：','').replace('reference:','').replace('。','')
        else:
            text=text.replace('reference：','').replace('reference:','').replace('。','')
        if '\n' in text:
            if text[:text.find('\n')]!='':
                print('a')
                text=text[:text.find('\n')].replace('\n','')
            else:
                text=text.replace('\n','')
    elif "改为" in text0:
        if "\n" in text0:
            continue
        else:
            text=text0[text0.rfind('改为'):][:text0.rfind('\n')].replace('“','').replace('\n','').replace('改为','').replace('”','')
    elif flag_text0==2:
        text=text00.replace('。','')
    if text==text00:
        flag=1
    print(temptext[-1])
    if text not in temptext[-1] and text not in black_list:
        dict_temp.setdefault(text) 
        dict_temp[text]=modelinput(present+'\n'+text)
    if flag_text0!=2:
        temptext[-1].append(text)
    elif text==max(dict_temp, key=lambda x: dict_temp[x]) and len(dict_temp)>=2:
        temptext[-1].append(text)
        flag_text0=0
    else:
        flag_text0=0
    temp.append(response["choices"][0]["message"]["content"])   
    number = Counter(temptext[-1])
    result = number.most_common()
    print("result01",result[0][1])
    print("result00",result[0][0])
    text00=text
    print(dict_temp)
    print(temptext[-1])
    if len(temptext[-1])>=12:
        if temptext[-1].count(max(dict_temp, key=lambda x: dict_temp[x]))>=2:
            save_csv(max(dict_temp, key=lambda x: dict_temp[x]),i,rname)
            data_output.append(max(dict_temp, key=lambda x: dict_temp[x]))
            dict_temp={}
            count0=0
            flag=0
            flag_text=0
            temptext.append([])
            i=i+1
            xr=0
            continue
        else:
            save_csv(result[0][0],i,rname)
            data_output.append(result[0][0])
            dict_temp={}
            count0=0
            flag=0
            flag_text=0
            temptext.append([])
            i=i+1
            xr=0
            continue
    if "无错误" in text0 or "无需修改" in text0 or "无误" in text0 or "不需要" in text0:
        if result[0][1]<2:
            flag=flag^1
            continue
        res=topn_dict(dict_temp,2)
        if result[0][0] not in res:
            flag=flag^1
            black_list.append(result[0][0])
            if result[0][0] in dict_temp:
                dict_temp.pop(result[0][0])
            while result[0][0] in temptext[-1]:
                temptext[-1].remove(result[0][0])
            text00=max(dict_temp, key=lambda x: dict_temp[x])
            continue
        else:
            save_csv(result[0][0],i,rname)
            data_output.append(result[0][0])
            dict_temp={}
            count0=0
            flag=0
            flag_text=0
            temptext.append([])            
            black_list=[]
            xr=0
            i=i+1
            continue
    elif result[0][1]==2 and result[0][0]==max(dict_temp, key=lambda x: dict_temp[x]):
        save_csv(result[0][0],i,rname)
        data_output.append(result[0][0])
        dict_temp={}
        count0=0
        flag=0
        flag_text=0
        black_list=[]
        temptext.append([])
        xr=0
        i=i+1
        continue        
    elif result[0][1]==3:
        res=topn_dict(dict_temp,2)
        print('最大值',res[0])
        if result[0][0] not in res:
            flag=flag^1
            black_list.append(result[0][0])
            if result[0][0] in dict_temp:
                dict_temp.pop(result[0][0])
            while text in temptext[-1]:
                temptext[-1].remove(text)
            text00=max(dict_temp, key=lambda x: dict_temp[x])
            continue
        else:
            save_csv(result[0][0],i,rname)
            data_output.append(result[0][0])
            dict_temp={}
            count0=0
            flag=0
            flag_text=0
            black_list=[]
            temptext.append([])
            xr=0
            i=i+1
            continue
    else:
        continue