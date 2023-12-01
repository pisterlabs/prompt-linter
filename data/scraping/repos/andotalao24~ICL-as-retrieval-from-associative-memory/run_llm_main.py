import copy
import time
import logging
from math import inf
#from accelerate import Accelerator
#from accelerate.logging import get_logger
import openai
import sys
import json
import random
import os
from keys import API_KEY
import math
import logging
import numpy as np
from utils import *
import argparse
from reformat import appendAns
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#from retrieval_method import getTopBM25, getTopSent

from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def is2Long(exemps, cot=1,max_len=2500): #4097-1024
    sent=''
    if len(exemps)==0:
        return False
    if 'data' in exemps[0]:
        exemps=[d['data'] for d in exemps]
    for d in exemps:
        if cot:
            try:
                sent+=d['que']+d['cot']+d['ans']
            except:
                n=random.randint(0,len(d['cot'])-1) if len(d['cot'])>1 else 0
                sent+=d['que']+d['cot'][n]+d['ans']
        else:
            sent+=d['que']+d['ans']
    return lenToken(sent) >= max_len


def lenToken(sent):
    return len(tokenizer.tokenize(sent))
#input_file = sys.argv[1].replace("\\", "\\")
#logger = get_logger(__name__)


def rank_select_exemp(exemp_file,dt):
    cmp=[(i,len(d['cot'].split())) for i,d in enumerate(dt)]
    cmp=sorted(cmp,key=lambda item:item[1],reverse=True)
    new_d=[dt[t[0]] for t in cmp]
    store_row(exemp_file,new_d)
    assert os.path.isfile(exemp_file)


def getExemplar(allData,k,cot=0):
    if cot:
        allData=[d for d in allData if d['cot']!=''] #if the dataset contains CoT
    #random.shuffle(allData)
    ret=random.sample(allData,k)
    tmp=k
    while is2Long(ret,cot):
        ret=random.sample(allData,tmp)
        tmp-=1
    return ret


def getAllExemplar(dt,k,l,cot=0):
    #k exemplar in one prompt
    #l length of input dt list
    if cot:
        try:
            dt=[d for d in dt if d['cot']!='' and d['exp']!=None] #if the dataset contains CoT
        except:
            dt = [d for d in dt ]
    random.shuffle(dt)
    ret=[random.sample(dt,k) for i in range(l)]

    for i,exemps in enumerate(ret):
        tmp=k
        while is2Long(exemps,cot):
            exemps=random.sample(dt,tmp)
            tmp-=1
        ret[i]=exemps

    return ret


def getExemplarRetrieval(train_dt,eval_dt,k,use_retrieval,cot=0):
    #use_retrieval: BM25 or sent
    if cot:
        try:
            train_dt=[d for d in train_dt if d['cot']!='' and d['exp']!=None] #if the dataset contains CoT
        except:
            train_dt = [d for d in train_dt ]
    if use_retrieval=='bm25':
        exemps=getTopBM25(eval_dt,train_dt,k)
    elif use_retrieval=='sbert':
        exemps=getTopSent(eval_dt,train_dt,k)
    else:
        print('wrong retrieval method, use random instead')
        exemps=getAllExemplar(train_dt,k,len(eval_dt),cot)
    return exemps


def getInps(multi_choice,change_ch,do_multi,multi_exp_idx,useJargon,role,train_dt,eval_dt,k,cutoff=0,dorank=0,diffExemp=0,cot=0,do_zero_cot=0,model='codex',exempFile='',use_retrieval='',shuffle=1,analysis_mode=0):

    prompt = ''
    #return raw data, selected from training st
    if analysis_mode:
        pass
    elif dorank:
        if diffExemp==0:
            exemps=train_dt[:k]
            if shuffle:
                random.shuffle(exemps)
        else:
            exemps=getAllExemplar(train_dt[:2*k],k,len(eval_dt),cot)
    elif use_retrieval !='':
        exemps=getExemplarRetrieval(train_dt,eval_dt,k,use_retrieval,cot)
    elif cutoff:
        assert k==1 and diffExemp==1
        exemps=[[train_dt[i]] for i in range(len(train_dt))]
    else:
        if diffExemp:
            exemps = getAllExemplar(train_dt, k, len(eval_dt), cot)
        else:
            exemps = getExemplar(train_dt, k, cot)
            if shuffle:
                random.shuffle(exemps)

    if model=='chat':
        message={"role": "system", "content": role}
        inputs=[]
        sample_qa=[]

        if not diffExemp:
            for ent in [chat_form(d, cot,do_multi,multi_exp_idx,i,multi_choice) for i,d in enumerate(exemps)]:
                sample_qa.extend(ent)

        for i,d in enumerate(eval_dt):
            t=[]
            t.append(message)
            if diffExemp:
                sample_qa = []
                for ent in [chat_form(d, cot,do_multi,multi_exp_idx,i,choice=multi_choice) for i,d in enumerate(exemps[i])]:
                    sample_qa.extend(ent) #extend
                t.extend(sample_qa) # sample qa is a list containing two dictionaries
            else:
                t.extend(sample_qa)

            if do_zero_cot and cot==1:
                t.append({'role': 'user', 'content': d['que']+d['cot']+'Therefore, the answer is, following the template {option index}:{answer span}. '})
            else:
                if k==0:
                    cue='Let\'s think step by step. '
                else:
                    cue=''
                t.append({'role': 'user', 'content': d['que']+cue})
            inputs.append(t)
    else:
        inputs=[]

        if analysis_mode:
            train_dt=train_dt['result']
            for i,d in enumerate(train_dt):
                inp = train_dt[i]['pred'][0]
                splitted=inp.strip().split(':')
                try:
                    if change_ch in ['A','B','C','D']:
                        splitted[-2]=splitted[-2].strip()[:-1]+change_ch  #change this line when calculating the prob of each option
                        inp = ':'.join(splitted[:-1])
                    pass
                except:
                    pass

                inputs.append(inp+ "\n")
            return inputs

        for i,d in enumerate(eval_dt):
            if k==0:
                cue=' Output the answer following the template. {option index}:{answer span}. '
                if useJargon:
                    inputs.append(prompt + 'Question: ' + d['que'] +d['jargon']+ cue)
                else:
                    inputs.append(prompt + 'Question: ' + d['que'] + cue)
            elif diffExemp:
                prompt = ''
                for ent in [dict2prompt(d, cot,do_multi,multi_exp_idx,j) for j,d in enumerate(exemps[i])]: # exemplars
                    prompt += ent
                inputs.append(prompt + 'Question: ' + d['que'] + "\nAnswer: ")
            else:
                prompt=''
                for ent in [dict2prompt(d, cot,do_multi,multi_exp_idx,j) for j,d in enumerate(exemps)]: # exemplars
                    prompt += ent
                inputs.append(prompt + 'Question: ' + d['que'] + "\nAnswer: ")
    return inputs


def run_eval(multi_choice,change_ch,analysis_mode,do_multi,multi_exp_idx,do_self_const,engine,useJargon,role,model,train_dt,eval_dt,output_file,use_retrieval,dorank=0,diffExemp=0,do_zero_cot=0,k=5,batch=10,cot=1,left=0,right=1000,cutoff=0,m=100):

    openai.api_key = API_KEY

    eval_dt = eval_dt[left:right] if right < len(eval_dt) else eval_dt[left:]
    if cutoff:
        train_dt = train_dt[left:right] if right < len(train_dt) else train_dt[left:]
    lenAll = len(eval_dt)
    golds=eval_dt #extraction of true ans is done during accuracy calculation
    '''
    train_dt -> exemplars -> prompt
    eval_dt -> input question 
    '''

    inputs=getInps(multi_choice,change_ch,do_multi,multi_exp_idx,useJargon,role,train_dt,eval_dt,k,cutoff,dorank,diffExemp,cot,do_zero_cot,model,use_retrieval,analysis_mode=analysis_mode)
    numBatch = math.ceil(lenAll / batch)
    print('len of eval: {}'.format(lenAll))
    print('num of batches: {}'.format(numBatch))
    skipBatch=[]
    preds,accs,ans=[],[],[]
    golds_store,inputs_store=[],[]
    probs_store=[]
    for i in range(numBatch):
        inputs_local = inputs[i * batch:(i + 1) * batch] if (i + 1) * batch <= lenAll else inputs[i * batch:lenAll]
        golds_local = golds[i * batch:(i + 1) * batch] if (i + 1) * batch <= lenAll else golds[i * batch:lenAll]
        if model=='text':
            kwargs = dict(engine=engine,
                          prompt=inputs_local,
                          temperature=0.7,
                          max_tokens=1024,
                          top_p=0.9,
                          frequency_penalty=0,
                          presence_penalty=0,
                          logprobs=1,
                          echo=False,
                          #stop='\n',
                          n=1
                          )
        elif model=='chat':
            kwargs= dict( model=engine,
                      messages=inputs_local,
                      max_tokens=1024,
                      temperature=0,
                          top_p=1,)

        async def run(**kwargs):
            repeat = 5
            flag = 0
            skip = 0
            response=[]
            probs=[]
            c=0
            if do_self_const:
                success = 5 # do 5 times sampling
                kwargs['temperature'] = 0.7
                repeat*=success
            else:
                success = 1
                kwargs['temperature']=0

            while flag<success:
                try:
                    print('requesting OpenAi')
                    c+=1
                    tmpresponse,prob=await requestResponse(model,**kwargs) #list of list
                    flag +=1
                    response.append(tmpresponse)

                    if kwargs['logprobs']>=1:
                        assert success==1
                        probs=prob


                except openai.error.RateLimitError as e:
                    print(e)
                    time.sleep(30)
                except Exception as e:
                    if 'maximum context length' in str(e):
                        print(e)
                        skip = 1
                        break
                    time.sleep(60)
                    print(e)
                if c==repeat:
                    time.sleep(70)
                if c>repeat:
                    if len(response)==0:
                        skip=1
                    break
            if do_self_const:
                print('self consistency check num of response {}'.format(len(response)))
            return response,skip,probs

        response,skip,probs=asyncio.run(run(**kwargs))


        if skip:
            skipBatch.append(i)
            continue
        preds_local = []


        for ii in range(len(response[0])): #batch size
            preds_local.append([response[j][ii] for j in range(len(response))])

        try:
            _,acc,anstmp=cal_acc(preds_local,golds_local,multi_choice)
        except Exception as e:
            print(e)
            acc=0
            anstmp=[]

        probs_store.extend(probs)
        ans.extend(anstmp)
        accs.append(acc)
        preds.extend(preds_local)
        golds_store.extend(golds_local)
        inputs_store.extend(inputs_local)
        appendResult(output_file.replace('.json','_cont.json'),preds_local,\
                     golds_local,inputs_local,acc)
        print(('eval, avg_acc of batch {}: {:.3f}').format(i, acc))
    if model=='text':
        with open(output_file.replace('.json','_prob.json'), 'w') as f:
            for p,acc in zip(probs_store,ans):
                f.write(json.dumps(p)+'\n')

    writeResult(output_file, preds, golds_store, inputs_store,accs)

    avg = np.average(np.array(accs).reshape(-1))
    print(('eval, total avg accuracy: ' + str(avg)))
    print('skipped batches: {}'.format(skipBatch))
    return avg



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--data',default=1,type=int)
    parser.add_argument('--left', default=0, type=int)
    parser.add_argument('--right', default=10000, type=int)
    parser.add_argument('--k', default=1, type=int)
    parser.add_argument('--m', default=2000, type=int)
    parser.add_argument('--batch', default=10, type=int)
    parser.add_argument('--select_cot',default=0,type=int)
    parser.add_argument('--do_cot', default=0, type=int)
    parser.add_argument('--do_zero_cot', default=1, type=int)
    parser.add_argument('--out',default='5-way-self-exp-amboss',type=str)
    parser.add_argument('--mode', default=3, type=int)
    parser.add_argument('--diffExemp', default=1, type=int)
    parser.add_argument('--model', default='chat', type=str)
    parser.add_argument('--dorank', default=0, type=int)
    parser.add_argument('--train_file', default='', type=str)
    parser.add_argument('--eval_file', default='', type=str)
    parser.add_argument('--use_retrieval', default='', type=str)
    parser.add_argument('--role', default='', type=str)
    parser.add_argument('--cutoff', default=1, type=int)
    parser.add_argument('--useJargon', default=0, type=int)
    parser.add_argument('--engine', default='gpt-3.5-turbo', type=str)
    parser.add_argument('--do_self_const', default=0, type=int)
    parser.add_argument('--do_multi', default=0, type=int)
    parser.add_argument('--multi_exp_idx', default=0, type=int)
    parser.add_argument('--do_analyze', default=0, type=int)
    parser.add_argument('--change_ch', default='NA', type=str)
    parser.add_argument('--multi_choice', default=1, type=int)
    parser.add_argument('--logprobs', default=1, type=int)
    args = parser.parse_args()
    params = vars(args)


    if params['train_file']!='':
        try:
            train_dt=read_row(os.path.join(BASE_DIR,params['train_file']))
        except:
            train_dt=read_Json(os.path.join(BASE_DIR,params['train_file']))
    if params['eval_file']!='':
        eval_dt=read_row(os.path.join(BASE_DIR,params['eval_file']))
        print(len(eval_dt))
    m=params['m']
    do_cot=params['do_cot']
    left=params['left']
    right=params['right']
    batch=params['batch']
    k=params['k']

    output_file= os.path.join(BASE_DIR,'bionlp/out/outputs/{}.json'.format(params['out']))
    run_eval(params['multi_choice'],params['change_ch'],params['do_analyze'],params['do_multi'],params['multi_exp_idx'],params['do_self_const'],params['engine'],params['useJargon'],params['role'],params['model'],train_dt,eval_dt, output_file,params['use_retrieval'],params['dorank'],params['diffExemp'],params['do_zero_cot'],k=k,batch=batch,cot=do_cot,left=left,right=right,m=m\
             ,cutoff=params['cutoff'])
    

if __name__ == '__main__':

    main()
