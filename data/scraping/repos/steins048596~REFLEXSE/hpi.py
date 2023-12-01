# -*- coding:utf-8 -*-
import os
#os.environ['http_proxy']="http://127.0.0.1:7892"
#os.environ['https_proxy']="http://127.0.0.1:7892"
import openai
openai.api_key = "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"
import csv
from collections import Counter
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from heapq import nlargest
import simplejson as json

import torch
import torch.nn as nn
from transformers import AutoTokenizer, LongformerModel
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional
import os
from hpi_dia import dialogue_process
import csv
import json
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


if torch.cuda.is_available():
    dev = torch.device("cuda:0") 
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    print("Running on the CPU")

tokenizer =AutoTokenizer.from_pretrained('longformer_clinical/')
tokenizer.pad_token = tokenizer.eos_token
test_data=[]

def read_data(typename,file_path_ini,file_path_reject,file_path_dia):   
    index=[]
    chosen=[]
    reject=[]
#     file_path_chosen='data/test.json'
#     file_path_reject='data/test_ini.json'
#     file_path_dia='data/dialogue.csv'
    with open(file_path_reject, 'r', encoding='utf8') as f:
        data = json.load(f)
    with open(file_path_ini, 'r', encoding='utf8') as f:
        data2 = json.load(f)
    for i in range(len(data['data'])):
        if data['data'][i]["subjective"]["HISTORY OF PRESENT ILLNESS"]!="" and data2['data'][i]["subjective"]["HISTORY OF PRESENT ILLNESS"]!="":
            index.append(data['data'][i]['encounter_id'])
    for item in data['data']:
        if item['encounter_id'] not in index:
            continue
        else:
            reject.append(item["subjective"]["HISTORY OF PRESENT ILLNESS"])
    for item in data2['data']:
        if item['file'][:item['file'].rfind('-')].replace('-','') in index:
            chosen.append(item["subjective"]["HISTORY OF PRESENT ILLNESS"])
        else:               
            continue
    dialogue=dialogue_process(typename,index,reject,file_path_dia,tokenizer,700,1200)
    for i in range(len(index)):
        temp={'index':index[i],'prompt':dialogue[i],'chosen':chosen[i],'rejected':reject[i]}
        if typename=='train':
            train_data.append(temp)
        if typename=='val':
            valid_data.append(temp)
        if typename=='test':
            test_data.append(temp)
read_data('test','data/test.json','data/test_ini.json','data/dialogue.csv')




class LEDRM(nn.Module):
    """
    BLOOM Reward model.

    Args:
        pretrained (str): Pretrained model name or path.
        config (BloomConfig): Model config.
        checkpoint (bool): Enable gradient checkpointing.
    """

    def __init__(self,
                 pretrained: str = None,
                 checkpoint: bool = False) -> None:
        super().__init__()
        self.model =LongformerModel.from_pretrained('longformer_clinical/')
        self.value_head = nn.Linear(self.model.config.hidden_size, 1, 1)
        self.value_head.weight.data.normal_(mean=0.0, std=1 / (self.model.config.hidden_size + 1))
    def forward(self, sequences: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.model(sequences, attention_mask=attention_mask)
        last_hidden_states = outputs['last_hidden_state']
        values = self.value_head(last_hidden_states)[:, :-1]
        value =values.mean(dim=1).squeeze(1)    # ensure shape is (B)
        return value

model = LEDRM().to(dev)
#小模型路径
load_layer_state_dict = torch.load("littlemodel/xianbingshi_real.pdparams")
model.load_state_dict(load_layer_state_dict)

def modelinput(prompt,chosen):
    model.eval()
    chosen = prompt + chosen + tokenizer.eos_token
    
    chosen_token = tokenizer(chosen,
                     max_length=1700,
                     padding="max_length",
                     truncation=True,
                     return_tensors="pt")
    chosen_ids = chosen_token['input_ids'].squeeze(1).to(dev)
    c_mask = chosen_token['attention_mask'].squeeze(1).to(dev)
    results = model(chosen_ids,c_mask).item()
    return results

def topn_dict(d, n):
    return nlargest(n, d, key=lambda k: d[k])

def save_csv(total,i,rname):
    with open(rname, 'a', newline='') as file:
        writer = csv.writer(file)
        if i==0:
            writer.writerow(["id", "text",])
        writer.writerow([test_data[i]['index'],total])
rname='results/hpi.csv'

print("a")

data_output=[]
present_total=[]
text_total=[]
  
temp=[]

@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(10))
def completion_with_backoff(prompt1,prompt2,prompt3,icl1,icl2,input0,temperature):
    result=openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional doctor, please help me point out the ERROR in the following medical records and REVISE."},
                    {"role": "user", "content": prompt1+icl1+icl2+prompt2+prompt3+input0}
                    # {"role": "assistant", "content": "https://www.cnblogs.com/botai/"},
                ],
                temperature=temperature,
                max_tokens=900
            )
#    print(prompt1+icl1+icl2+prompt2+prompt3+input0)
    return result
#i=121
i=0
t=0
w=0
xr=0
tt=0
temptext=[]
black_list=[]
temptext.append([])
dict_temp={}
count0=0
flag=0
tempmax=''
flag_text=0
while(i<=len(test_data)):
#while(i<171):
    print("I:",i)
    if flag_text==0:
        text00=str(test_data[i]['rejected'])
    temperature=0.2
    print(text00)
    if flag_text==0:
        temptext[-1].append(text00)
        dict_temp.setdefault(text00)
        dict_temp[text00]=modelinput(str(test_data[i]['prompt']),str(test_data[i]['rejected']))
#    print(present)
    icl1="EXAMPLE NOTE：\nHISTORY OF PRESENT ILLNESS\nMr. Brian is a 58-year-old male with a past medical history significant for congestive heart failure and hypertension, who presents today for follow-up of his chronic problems. Mr. Brian has been feeling out of sorts, experiencing fatigue, and occasional lightheadedness for approximately five weeks, since Labor Day. He notes bloating, which may be associated with weight gain. The patient has been involved in home renovations leading to dietary changes, including eating out more frequently. He feels short of breath during physical exertion related to his projects and reports mild chest cramps that resolve in about an hour and a slight cough. His blood pressure has been well-controlled at home.\nEXAMPLE ERROR:\nCompared to the DIALOGUE, Some information was missing from the present medical history, such as, the patient is not sure whether the cough is due to seasonal changes, eating habits since last year, and medication for high blood pressure and denial of some symptoms.\nREVISE: \nBrian is a 58-year-old male with a past medical history significant for congestive heart failure and hypertension, who presents today for follow-up of his chronic problems.The patient states he has been feeling out of sorts lately. He is not sure if it is due to the change in the seasons or due to performing lots of projects and some construction on his home. He reports fatigue and lightheadedness. This has been going on for about 5 weeks. While exerting energy, he has experienced some shortness of breath and chest cramps. The patient also notes a slight cough, but he is not sure if it is just the change in seasons.He feels bloated every once in a while. His diet has been a little bit of a struggle. They had construction on their kitchen begin over Labor Day weekend, and have been eating less healthy food as a result. Regarding his heart failure, he has been pretty good with his salt intake. He has been pretty good about his diet since the last year and is staying on top of that as much as possible.The patient has continued to utilize Lasix daily.He has continued to monitor his blood pressure regularly.He denies weight gain, swelling in the lower extremities, fevers, chills, dizziness, nausea, vomiting, and diarrhea.\n\n"
    icl2="EXAMPLE NOTE：\nHISTORY OF PRESENT ILLNESS\nBrittany is a patient who presented to the clinic with an injury to her right foot sustained a couple of days ago while playing tennis. During a doubles game, she fell on top of her foot while attempting to volley the ball, resulting in pain. She was unable to continue playing and needed assistance leaving the field. The patient reports wrapping and icing her foot and taking ibuprofen for pain relief, but the pain persists. Brittany has previously injured her left foot but has no history of injury to the right foot.\nEXAMPLE ERROR:\nCompared to the DIALOGUE, there are many ambiguities in HISTORY OF PRESENT ILLNESS, such as the patient's right foot being injured \"two days ago\" instead of \"a few days ago\", and missing records, the patient denies numbness and loss of sensation.\nRevise:\nBrittany is a right-hand-dominant female who presents to the clinic today for the evaluation of right foot pain. The onset of her pain began 2 days ago, when she was playing tennis and was trying to volley the ball when she got in front of another player and fell on the dorsal aspect of her right foot. She states that she quickly twisted her foot because she was trying to catch herself. The patient reports that she was unable to continue playing secondary to the pain. She states that she wrapped her foot after the game and iced it last night. The patient adds that she kept her foot up on a pillow and took ibuprofen for pain.She denies any numbness. The patient denies any loss of sensation.The patient has a history of a left leg injury.\n\n"
    prompt1="GUIDE:\nThe HISTORY OF PRESENT ILLNESS is the central component of this history and should:\nProvides a chronological description of the patient’s story, usually addressing the story of the symptoms first.  (manifestations/symptoms of illness, interventions, patient interpretations).\nIncludes pertinent context from the “rest of the history”.  All historical information that is key to understanding the differential diagnosis of the primary problem(s) should be included here. The HPI should be written in prose with full sentences and be a narrative that builds an argument for the reason the patient was admitted.Do not write inspection indicators and doctor's recommendations into HISTORY OF PRESENT ILLNESS\n\n"
    prompt2="----------------------------------------------------------------------------------------------\n\nDIALOGUE：\n"+str(test_data[i]['prompt'])+'\n'
    prompt3="Now, referring to the GUIDE, recorrect the NOTE with the DIALOGUE and refer the EXAMPLE above.DO NOT REVISE THE EXAMPLE.\n\n"
    input0="NOTE：\nHISTORY OF PRESENT ILLNESS\n"+text00+"\n\nERROR:\n"

    # 调用 ChatGPT
    response = completion_with_backoff(prompt1,prompt2,prompt3,icl1,icl2,input0,temperature)
    text0=str(response["choices"][0]["message"]["content"])
    print(text0)
    flag_text=1
    if "REVISE:" not in text0:
        print('bbbb')
        continue
#     w=0
    if "REVISE:" in text0:
        text=text0[text0.rfind('REVISE:'):][:text0.rfind('。')].replace('REVISE:','').replace('\n','')
        if '\n' in text:
            print("test",text[:text.find('\n')])
            if text[:text.find('\n')]!='':
                print('a')
                text=text[:text.find('\n')].replace('\n','')
            else:
                text=text.replace('\n','')
    print(temptext[-1])
    print('小模型输入',text)
    if 'numbness' in text00 and 'loss of sensation' in text00:
        ff=1
    else:
        if 'numbness' in text and 'loss of sensation' in text:
            print('cccc')
            t=t+1
            continue
    if 'left leg injury' in text00:
        ff=1
    else:
        if 'left leg injury' in text:
            print('cccc')
            t=t+1
            continue
    if '58' in text00:
        ff=1
    else:
        if '58-year-old' in text:
            print('cccc')
            t=t+1
            continue
    if 'Brian' in text00:
        ff=1
    else:
        if 'Brian' in text:
            print('cccc')
            t=t+1
            continue
    if 'Brittany' in text00:
        ff=1
    else:
        if 'Brittany' in text:
            print('cccc')
            t=t+1
            continue
    if text in black_list:
        flag=flag^1
        continue
    if text not in dict_temp.keys() and text not in black_list:
        dict_temp.setdefault(text) 
        dict_temp[text]=modelinput(str(test_data[i]['prompt']),text)
    
    temptext[-1].append(text)
    print('结果',text)
    temp.append(response["choices"][0]["message"]["content"])   
    number = Counter(temptext[-1])
    result = number.most_common()
    print("result01",result[0][1])
    print("result00",result[0][0])
    tempmax=result[0][0]
    text00=text
    print(dict_temp)
    print(temptext[-1])
    if len(temptext[-1])>=16:
        if temptext[-1].count(max(dict_temp, key=lambda x: dict_temp[x]))>=2 or max(dict_temp, key=lambda x: dict_temp[x])==temptext[-1][-1]:
            save_csv(max(dict_temp, key=lambda x: dict_temp[x]),i,rname)
            data_output.append(max(dict_temp, key=lambda x: dict_temp[x]))
            dict_temp={}
            count0=0
            flag=0
            flag_text=0
            temptext.append([])
            i=i+1
            xr=0
            t=0
            tempmax=''
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
            t=0
            tempmax=''
            continue
            
    elif result[0][1]>=3:
        res=topn_dict(dict_temp,2)
        print('最大值',res[0])
        if len(dict_temp)<=3:
            flag=flag^1
            continue
        if result[0][0] not in res:
            flag=flag^1
            black_list.append(result[0][0])
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
            black_list=[]
            temptext.append([])            
            i=i+1
            xr=0
            tempmax=''
            t=0
            continue
