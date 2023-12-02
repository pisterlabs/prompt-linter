# -*- coding:utf-8 -*-
import paddle
paddle.device.set_device('gpu:3')
import os
#os.environ['http_proxy']="http://127.0.0.1:7892"
#os.environ['https_proxy']="http://127.0.0.1:7892"
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
load_layer_state_dict = paddle.load("little_model/real_model/xianbingshi.pdparams")
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
rname='new_experiment/hpi_chatgpt_fewshot.csv'

data_output=[]
present_total=[]
text_total=[]
filename='data/test_index.csv'
#filename='../new_experiment/16ktest/index_test.csv'
with open(filename,encoding='gbk') as f:
        i=0    
        f_csv=csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            text_total.append(str(row[7]))
#            text_total.append(str(row[2]))
            i=i+1
temp=[]
@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(10))
def completion_with_backoff2(prompt1,prompt2,prompt3,icl1,icl2,input0,temperature):
    result=openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=[
                    {"role": "system", "content": "你是一位专业医生，请根据具体的规则和要求修改电子病历，并生成Errorprompt和reference"},
                    {"role": "user", "content": prompt1+prompt3+icl1+icl2+prompt2+input0}
                    # {"role": "assistant", "content": "https://www.cnblogs.com/botai/"},
                ],
                temperature=temperature
            )
#    print(prompt1+icl1+icl2+prompt2+prompt3+input0)
    return result
@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(10))
def completion_with_backoff(prompt1,prompt2,prompt3,icl1,icl2,input0,temperature):
    result=openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=[
                    {"role": "system", "content": "你是一位专业医生，请根据具体的规则和要求修改电子病历，并生成Errorprompt和reference"},
                    {"role": "user", "content": prompt1+icl1+icl2+prompt2+prompt3+input0}
                ],
                temperature=temperature
            )
#    print(prompt1+icl1+icl2+prompt2+prompt3+input0)
    return result
#i=121
i=0
t=0
w=0
xr=0
temptext=[]
black_list=[]
temptext.append([])
dict_temp={}
count0=0
flag=0
tempmax=''
flag_text=0
#lack_list=[1,2,4,7,26,52,100,103,124,125,150,199]
lack_list=[124]
while(i<=len(text_total)):
#while(i<171):
    print("I:",i)
# 设置 API 密钥
    if flag_text==0:
        text00=str(text_total[i]).replace('(2)现病史：','')
    temperature=0.2
    print(text00)
    if flag_text==0:
        temptext[-1].append(text00)
        dict_temp.setdefault(text00)
        dict_temp[text00]=modelinput(str(text_total[i]).replace('(2)现病史：',''))
#    print(present)
    icl1="(2)现病史：患者昨天发烧，但退烧药后退烧。今天喉咙有充血和红点，扁桃体有三四个白色小泡，无咳嗽和手足包。口服头孢类药物治疗，发烧症状未超过38度，低烧37.6度。无化脓点。\n\nErrorPrompt：\n字数不能过多，症状描述要简练，今天”“昨天”用“一（两）天前”代替 \n\nreference：\n患儿一天前无明显诱因下出现发烧伴扁桃体白色小泡情况，无咳嗽咳痰，无恶心呕吐，无其他明显不适症状。精神状态一般，胃纳一般，余如常。\n\n"
    icl2="(2)现病史：宝宝最近一周大便3-4次，无受凉史。建议查大便常规，给宝宝多吃一些白粥米汤和益生菌类食物，少喝一些奶粉，如需喝奶粉可以试试乳糖酶，大型母婴店可购买。\n\nErrorPrompt：\n简短精炼，不需要描述病程细节，禁止出现建议，禁止口语化\n\nreference：\n患儿腹泻，每天3-4次。\n\n"
    prompt1="规则：现病史 是指患者 本次疾病的发生（按时间顺序）、演变、诊疗等 方面的详细情况，应当和主诉一致，描写要确切，用词要恰当，精炼。\n1.发病情况：记录发病的时间、地点、起病缓急、可能的原因或诱因。\n2.主要症状特点及其发展变化情况：按发生的先后顺序描述主要症状的部位、性质（持续性，阵发性）、持续时间、程度、缓解（进行性加剧，逐渐好转）或加剧因素（劳动，姿态等），以及演变发展情况。 \n3.伴随症状：记录伴随症状（发热，流汗，头痛，头晕，呕吐 ，休克等），描述伴随症状与主要症状之间的相互关系。\n4.发病以来诊治经过及结果：记录患者发病后到接受检查与治疗的详细经过及效果。包括患者诉说的药名、诊断和手术名称。\n5.发病以来一般情况：简要记录患者发病后的精神状态、睡眠、食欲、大小便、体重等情况。\n6.不需要写“建议”和既往史相关内容\n7.禁止口语化，简短精炼。\n参考上述要求 \nErrorPrompt一定要简练\n "
#    if len(str(text_total[i]).replace('(1)主诉：',''))>10:
    prompt2="“提示”："+str(text_total[i]).replace('(2)现病史：','')+'\n'
#    else:
#        prompt2="“提示”："+present+'\n'
    prompt3="参考规则，生成ErrorPrompt逐条列出下方“(2)现病史”的错误，并在\"reference\"中修改Errorprompt列出的错误，并对比“提示”和下方“现病史”，\"reference\"中禁止生成“提示”以外的信息,简短精炼\n"
    if flag==0:
        input0="(2)现病史："+text00+"\n\nErrorPrompt："
    else:
        input0="(2)现病史："+text00+"\n\n"

    if flag==0:
        if t<3:
            print(1)
            response = completion_with_backoff(prompt1,prompt2,prompt3,icl1,icl2,input0,temperature)
        else:
            print(2)
            response = completion_with_backoff2(prompt1,prompt2,prompt3,icl1,icl2,input0,temperature)
    elif flag==1:
        if t<3:
            print(3)
            response = completion_with_backoff(prompt1+'\n',prompt2,prompt3,icl2,icl1,input0,temperature)           
        else:
            print(4)
            response = completion_with_backoff2(prompt1+'\n',prompt2,prompt3,icl2,icl1,input0,temperature)

    text0=str(response["choices"][0]["message"]["content"])
    print(text0)
    flag_text=1
    if "ErrorPrompt" not in text0 and "ErrorPrompt：" not in input0:
        t=t+1
        print('aaaaa')
        flag=flag^1
        continue
    if "reference" not in text0 and "改为" not in text0 and "Reference" not in text0:
        print('bbbb')
        if "无错误" not in text0 and "无需修改" not in text0 and "无误" not in text0:
            t=t+1
            continue
        else:
            flag_text=2
    w=0

    if "reference" in text0 or "Reference" in text0:
        t=0
        print("right")
        if "reference：" in text0:
            text=text0[text0.find('reference：'):].replace('reference：\n','').replace('reference：','').replace(' ','')
        elif "reference:" in text0:
            text=text0[text0.find('reference:'):].replace('reference:\n','').replace('reference:','').replace(' ','')
        elif "Reference：" in text0:
            text=text0[text0.find('Reference：'):].replace('Reference：\n','').replace('Reference：','').replace(' ','')
        elif "Reference:" in text0:
            text=text0[text0.find('Reference:'):].replace('Reference:\n','').replace('Reference:','').replace(' ','')
        if "。" in text:
            text=text[:text.rfind('。')]+'。'
        if '\n' in text:
            print("test",text[:text.find('\n')])
            if text[:text.find('\n')]!='':
                print('a')
                text=text[:text.find('\n')].replace('\n','')
            else:
                text=text.replace('\n','')
    elif "修改为以下格式：" in text0:
        text=text0[text0.rfind('修改为以下格式：'):][:text0.rfind('\n')].replace('“','').replace('\n','').replace('修改为以下格式：','').replace('”','')
        if "ErrorPrompt：" in text:
            text=text[:text.rfind('ErrorPrompt：')].replace('ErrorPrompt：','')
        if "提示：" in text:
            text=text[:text.rfind('提示：')].replace('提示：','')
    elif "请修改为：" in text0:
        text=text0[text0.rfind('请修改为：'):][:text0.rfind('\n')].replace('“','').replace('\n','').replace('请修改为：','').replace('”','')
        if "ErrorPrompt：" in text:
            text=text[:text.rfind('ErrorPrompt：')].replace('ErrorPrompt：','')
        if "提示：" in text:
            text=text[:text.rfind('提示：')].replace('提示：','')
    elif "改为" in text0:
        text=text0[text0.rfind('改为'):][:text0.rfind('\n')].replace('“','').replace('\n','').replace('改为','').replace('”','')
        if "ErrorPrompt：" in text:
            text=text[:text.rfind('ErrorPrompt：')].replace('ErrorPrompt：','')
        if "提示：" in text:
            text=text[:text.rfind('提示：')].replace('提示：','')
        if "“提示”：" in text:
            text=text[:text.rfind('“提示”：')].replace('“提示”：','')
    
    elif flag_text==2:
        text=text00
        flag_text=1
#    print(temptext[-1])
    print(temptext[-1])
    print('小模型输入',text)
    if '3-4' not in text00:
        if '患儿腹泻，每天3-4次' in text:
            print('cccc')
            t=t+1
            continue

    if text==tempmax and temptext[-1].count(text)>=2:
        flag=flag^1
        continue
    if text in black_list:
        flag=flag^1
        continue
    if text not in dict_temp.keys() and text not in black_list:
        dict_temp.setdefault(text) 
        dict_temp[text]=modelinput(text)
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
    if len(temptext[-1])>=18:
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
    if "无错误" in text0 or "无需修改" in text0 or "无误" in text0:
        if temptext[-1].count(max(dict_temp, key=lambda x: dict_temp[x]))<5:
            if len(dict_temp)<3:
                flag=flag^1
                continue
        res=topn_dict(dict_temp,2)
        if result[0][0] not in res:
            flag=flag^1
            black_list.append(result[0][0])
#             if result[0][0] in dict_temp:
#                 dict_temp.pop(result[0][0])
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
            i=i+1
            xr=0
            tempmax=''
            t=0
            continue
            
    elif result[0][1]>=2:
        res=topn_dict(dict_temp,2)
        if len(dict_temp)<=3:
            if result[0][0]==str(text_total[i]).replace('(2)现病史：',''):
                text00=temptext[-1][temptext[-1].index(result[0][0])-1]
            flag=flag^1
            continue
        if result[0][0] not in res:
            flag=flag^1
            black_list.append(result[0][0])
#             if result[0][0] in dict_temp:
#                 dict_temp.pop(result[0][0])
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