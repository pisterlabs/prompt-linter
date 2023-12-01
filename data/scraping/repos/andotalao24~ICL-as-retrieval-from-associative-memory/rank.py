import time
from math import inf
from utils import *
import openai
import json
import random
import os
from keys import API_KEY
import math
import numpy as np
import argparse
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def getExemplarPool(input_json,dataset):
    #m: pool size
    #input_json: training data
    fileName=input_json.replace('.json','Pool_exemplar.json')
    if dataset == 1:
        allData = process_medcq(input_json)
    elif dataset == 0:
        allData = process_usmle(input_json)
    else:
        try:
            allData=read_row(input_json)
        except:
            allData=read_Json(input_json)

    if not os.path.exists(fileName):
        print('exemplar pool not found. start generating')
        random.shuffle(allData)
        with open(fileName,'w') as f:
            json.dump([dt for dt in allData],f,indent=2)
    else:
        print('found exemplar pool')

    return allData


def eval(model,inputs, golds, batch=10, outputFile=None):
    openai.api_key = API_KEY

    lenAll = len(inputs)
    print(lenAll)
    numBatch = math.ceil(lenAll / batch)
    avgAcc = []
    print('num of batches: {}'.format(numBatch))
    for i in range(numBatch):
        inputs_local = inputs[i * batch:(i + 1) * batch] if (i + 1) * batch <= lenAll else inputs[-batch:-1]
        golds_local = golds[i * batch:(i + 1) * batch] if (i + 1) * batch <= lenAll else golds[-batch:-1]

        if model=='codex':
            kwargs = dict(engine="text-davinci-003",
                          prompt=inputs_local,
                          # use batching by passing a list of prompts. You can use up to 20 in a batch.
                          temperature=0.0,
                          max_tokens=1024,
                          top_p=1,
                          frequency_penalty=0,
                          presence_penalty=0,
                          stop='\n',
                          n=1
                          )
        elif model=='chat':
            kwargs= dict( model="gpt-3.5-turbo",
                      messages=inputs_local,
                      max_tokens=1024,
                      temperature=0)

        async def run():
            flag = 1
            response=[]
            while flag:
                try:
                    print('requesting OpenAi')
                    response = await requestResponse(model,**kwargs)
                    flag = 0
                except Exception as e:
                    print(e)
                    time.sleep(20)
            return response

        response = asyncio.run(run())

        preds_local = []
        for pred in response:
            preds_local.append(pred)

        _,acc = cal_acc(preds_local, golds_local)
        avgAcc.append(acc)
        print(('eval, avg acc of batch {}: {:.3f}').format(i, acc))

    if outputFile != None:
        pass

    avg = np.average(np.array(avgAcc).reshape(-1))
    print(('eval, total acc: ' + str(avg)))
    return avg


def rankGreedy(model, exempFile, m,k,n,batch=1):
    if not os.path.exists(exempFile):
        print('ranked exemplars not found')
        return
    exemps=read_Json(exempFile)
    st=exemps[0]
    try:
        cot=st['data']['cot']
    except:
        cot=0
    assert model=='chat'
    assert m>n
    assert m==len(exemps)
    assert m>k


    sysIntr = {"role": "system", "content": "You are a physician to answer patients' questions."}
    retIdx=[0] #index of selected exemplars
    scorelist=[st['avgAcc']]
    for i in range(k-1):
        score=0
        winner=0
        for j in range(1,m):#skip the top
            #candidate exemplar
            if j in retIdx:
                continue
            exemplar = chat_form(exemps[j], cot)
            inputs = []
            golds = []
            tmpCmp=random.sample(range(1,m),n)
            for j in range(n):
                if tmpCmp[i] in retIdx or tmpCmp[i]==j:
                    continue
                message=[sysIntr]
                for idx in retIdx:
                    message.extend(chat_form(exemps[idx], cot))
                message.extend(exemplar)
                message.append(getQue_chat(exemps[tmpCmp[i]]['data']['que']))
                inputs.append(message)
                golds.append(exemps[tmpCmp[i]]['data'])

            assert len(inputs) == len(golds)
            s_tmp = eval(model, inputs, golds, batch)
            if s_tmp>score:
                score=s_tmp
                winner=j
        scorelist.append(score)
        retIdx.append(winner)

    with open(exempFile.replace('.json','_seq_{}.json'.format(k)), 'w') as f:
        json.dump([{'data': exemps[retIdx[i]]['data'], 'avgAcc': scorelist[i]} for i in range(len(retIdx))], f, indent=2)




def rankExemplar(model,inputFile, exempFile, m,cot=0,dataset=0, batch=10):
    # rank based on other exemplars in the exemplar pool
    if os.path.exists(exempFile):
        print('ranked exemplars found')
        return
    print('ranked exemplars not found, start to rank')
    exemplarPool = getExemplarPool(inputFile,dataset)  # randomly get k ememplars
    n= len(exemplarPool)
    f1dic = dict()
    if model!='chat':
        #conduct self comparison in the pool whose size is m
        for i in range(n):
            #select the current exemplar
            exemplar = dict2prompt(exemplarPool[i],cot)
            inputs = []
            golds = []
            #test on other training cases
            idxlist=random.sample(range(n),m)
            for j in idxlist:
                if j == i:
                    continue
                inputs.append(exemplar + 'Question: ' + exemplarPool[j]['que'] + "\nAnswer: ")
                golds.append(exemplarPool[j])

            assert len(inputs) == len(golds)
            f1dic[i] = eval(model,inputs, golds, batch)  # f1
    else:
        # conduct self comparison in the pool whose size is m
        for i in range(n):
            # select the current exemplar
            message = {"role": "system", "content": "You are a physician to answer patients' questions."}
            exemplar = chat_form(exemplarPool[i], cot)
            inputs = []
            golds = []
            # test on other training cases
            idxlist = random.sample(range(n), m)
            for j in idxlist:
                if j == i:
                    continue
                inputs.append([message,exemplar[0],exemplar[1],getQue_chat(exemplarPool[j]['que'])])
                golds.append(exemplarPool[j])

            assert len(inputs) == len(golds)
            f1dic[i] = eval(model, inputs, golds, batch)  # f1

    ranked_idx = [k for k, v in sorted(f1dic.items(), key=lambda item: item[1], reverse=True)]

    with open(exempFile, 'w') as f:
        json.dump([{'data': exemplarPool[i], 'avgAcc': f1dic[i]} for i in ranked_idx], f, indent=2)



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--m',default=1,type=int)
    parser.add_argument('--batch', default=1, type=int)
    parser.add_argument('--cot', default=0, type=int)
    parser.add_argument('--mode', default=0, type=int)
    parser.add_argument('--model', default='chat', type=str)
    parser.add_argument('--type', default='single', type=str)
    parser.add_argument('--n', default=1, type=int)
    parser.add_argument('--k', default=1, type=int)
    parser.add_argument('--exempFile', default='', type=str)
    args = parser.parse_args()
    params = vars(args)


    m=params['m'] #pool size
    cot=params['cot'] #whether to do cot
    dataset=params['mode'] #use USMLE
    model=params['model']
    if dataset==0:
        inputFile = os.path.join(os.getcwd(), '../data/medQA/train.json')
    elif dataset==1:
        inputFile = os.path.join(os.getcwd(), '../data/medCQ/train.json')
    else:
        inputFile= os.path.join(BASE_DIR,params['exempFile'])

    exempFile = inputFile.replace('.json','ranked_exemplar_{}.json'.format(m))
    if params['type']=='seq':
        rankGreedy(model, exempFile, m, params['k'], params['n'])
    else:
        rankExemplar(model,inputFile,exempFile,m,cot,dataset,batch=1)


if __name__ == '__main__':
    main()
