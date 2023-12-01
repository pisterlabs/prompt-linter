#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 20:37:53 2022

@author: chengyu
"""

import openai
import os,sys
import numpy as np
from transformers import GPT2TokenizerFast
sys.path.insert(0,'../../lib')
from utils import load_json,load_jsonl
import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score,ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#%%

class GPT(object):
    
    def __init__(self,api_key):
        """
        initiate an openai instance 
        """
        self.openai = openai
        self.openai.api_key = api_key
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        
    def get_GPT3_file(self,f_name=None,id_name_only=True):
        """
        get uploaded files in openai server
        """
        ### get files 
        files = self.openai.File.list()['data']
        
        ### filter files 
        if f_name:
            files = [f for f in files if f['filename'] == f_name]
            if len(files)>1:
                files=[files[0]]
        else:
            pass
        
        ### keep simple 
        if id_name_only:
            files = [{'name':f['filename'],'id':f['id']} for f in files]
        
        ### return one object 
        if len(files)==1:
            files = files[0]
            
        return files

    def _classify(self,file_id,query,logprobs,labels,
                search_model="ada", 
                model="text-davinci-001",   #"text-davinci-001"; "curie"
                temperature=0.0,
                max_examples=30):
        
        result =self.openai.Classification.create(
            file=file_id,
            query=query,
            logprobs=logprobs,
            labels=labels,
            search_model=search_model, 
            model=model,   #"text-davinci-001"; "curie"
            temperature=temperature,
            max_examples=max_examples
            )
        
        return result
    
    def _get_probs(self,result,labels):
        """
        result : openai response from classify class
        labels : all classes available
        follow : https://beta.openai.com/docs/guides/classifications
        """
        
        labels = [label.strip().lower().capitalize() for label in labels]
        labels_tokens = {label: self.tokenizer.encode(" " + label) for label in labels}
        first_token_to_label = {tokens[0]: label for label, tokens in labels_tokens.items()}
        
        if "completion" in result.keys():
            top_logprobs = result["completion"]["choices"][0]["logprobs"]["top_logprobs"][0]
        elif "choices" in result.keys():
            top_logprobs = result["choices"][0]["logprobs"]["top_logprobs"][0]
        else:
            raise("error, result format problem!")
            
        token_probs = {
            self.tokenizer.encode(token)[0]: np.exp(logp) 
            for token, logp in top_logprobs.items()
        }
        label_probs = {
            first_token_to_label[token]: prob 
            for token, prob in token_probs.items()
            if token in first_token_to_label
        }
        
        # check in case lable is not in returned topn 
        for l in labels:
            if l not in label_probs.keys():
                label_probs[l] = 0.0
        
        # Fill in the probability for the special "Unknown" label.
        if sum(label_probs.values()) < 1.0:
            label_probs["Unknown"] = 1.0 - sum(label_probs.values())

            
        return label_probs
    
    def classify_predict(self,file_id,query,logprobs,labels,n_ref =0):
        ## get results 
        res = self._classify(file_id,query,logprobs,labels)
        ## get label only
        pre_label = res["label"]
        ## get probs
        probs = self._get_probs(res,labels)
        
        ## return top n references
        if n_ref>0:
            res_examples = res["selected_examples"]
            l = sorted(res_examples, key = lambda i: i['score'],reverse=True)
            refs = l[:n_ref]
        else:
            refs = []
            
        return pre_label,probs,refs

    def _simple_classify(self,prompt,engine="text-davinci-001",logprobs=4,temperature=0.0,labels=None):
        response = self.openai.Completion.create(
                                            engine=engine,
                                            prompt=prompt,
                                            logprobs=logprobs,
                                            temperature=temperature
                                            )
        
        pre_label = response['choices'][0]['text'].strip()
        
        if labels:
            prob = self._get_probs(response,labels)
        else:
            prob=None
            
        return pre_label,prob,response


def formate_tweet_prompt(query):
    task = "Decide whether a Tweet's sentiment is Positive, Neutral, or Negative."
    task = '{}\n\n'.format(task)
    q = """Tweet: {}\nSentiment:""".format(query)
    prompt = task + q 
    return prompt

def run_one_example(query):
    simple_prompt = formate_tweet_prompt(query)
    
    ## run with few shot setting
    pre_label,probs,refs= gpt.classify_predict(file_id=file_id,
                                                 query=query,
                                                 logprobs=logprobs,
                                                 labels=labels,
                                                 n_ref =1)
    print("\n\nResults from few shot setting:")
    print("  Predicted Label: {}\n  Predicted Prob: {}\n  Top Reference: {}".format(pre_label,probs,refs))
    
    ## run with zero shot 
    s_label,s_prob,s_res = gpt._simple_classify(simple_prompt,labels=labels)
    print("\n\nResults from zero shot setting:")
    print("  Predicted Label: {}\n  Predicted Prob: {}\n\n".format(s_label,s_prob))

#%%

if __name__ == '__main__':

    root = '../../../'
    key_path = os.path.join(root,'GPT_SECRET_KEY.json')
    data_folder= os.path.join(root,'data')
    train_file = os.path.join(data_folder,'tweets_train.jsonl')     ## will need to first upload file 
    test_file = os.path.join(data_folder,'tweets_test.jsonl')
    result_file = os.path.join(data_folder,'eval.xlsx')
    #%%
    gpt = GPT(api_key = load_json(key_path)['API_KEY'])
    
    ##### if training data is not upload it will need to upload first 
    ### use File api to upload traning data to server as jsonl format
    #file_id = gpt.openai.File.create(file=open(data_path), purpose="classifications")
    
    #%% ## set up all paramaters 
    file_id = gpt.get_GPT3_file(f_name ='tweets_train.jsonl')['id']
    labels = ["Positive","Neutral","Negative"]
    logprobs = len(labels) +1
    
    #%%
    ## try run one example 
    query = "so the bill is going from working on climate change paid family leave free community college to what a bunch of subsidies for large corporations"
    run_one_example(query)
    
    #%%
    ## evaluate on test file 
    train = load_jsonl(train_file)
    test = load_jsonl(test_file)
    for idx,t in enumerate(test):
        
        if idx % 20 == 0:
            print("........processing {}/{}........".format(idx,len(test)))
        
        query = t['text']
        ## run few shot 
        pre_label,probs,refs= gpt.classify_predict(file_id=file_id,
                                                     query=query,
                                                     logprobs=logprobs,
                                                     labels=labels,
                                                     n_ref =1)
        t['f_pre_label'] = pre_label
        t['f_probs'] = probs
        t['f_ref'] = refs
        ## run zero shot
        simple_prompt = formate_tweet_prompt(query)
        s_label,s_prob,s_res = gpt._simple_classify(simple_prompt,labels=labels)
        t['z_pre_label'] = s_label
        t['z_probs'] = s_prob
    
    #%%
    res_test = test[:141]
    
    #%%
    ### get evaluation results 
    df = pd.DataFrame(res_test)
    df.to_excel(result_file)
    
    ### 
    f_accuracy = accuracy_score(df['label'], df['f_pre_label'])
    print('Few shot accuarcy: {}'.format(f_accuracy))
    conf = ConfusionMatrixDisplay.from_predictions(df['label'], df['f_pre_label'])
    plt.show()
    
    z_accuracy = accuracy_score(df['label'], df['z_pre_label'])
    print('Zero shot accuarcy: {}'.format(z_accuracy))
    conf = ConfusionMatrixDisplay.from_predictions(df['label'], df['z_pre_label'])
    plt.show()
    #%%
    