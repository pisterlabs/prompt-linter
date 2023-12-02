## Get GPT3 responses for all SG papers

import os
import sys
import openai
import pandas as pd
import numpy as np
import argparse
from transformers import BloomTokenizerFast, BloomModel,BloomForCausalLM
import torch
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

import transformers

openai.api_key = os.getenv("OPENAI_API_KEY")

def main():
    data_path="../data/"
    outputs_path="../outputs/"
    positives=pd.read_csv(outputs_path+"sg_ie/positives_ready.csv")

    df=positives.reset_index(drop=True).copy()

    preprompt="There is an NLP paper with the title and abstract:\n"
    question="Which of the UN goals does this paper directly contribute to? Provide the goal number and name."
    df=df.assign(statement=preprompt+df.title_abstract_clean+"\n"+question)

    for i,d in df.iloc[i:,:].iterrows():
        input_prompt=d['statement']
        completion = openai.Completion.create(engine="text-davinci-002", prompt=input_prompt,temperature=0,max_tokens=50,logprobs=1)
        dict_norm={}
        dict_uniques={}

        df.loc[i,'full_prompt']=input_prompt
        df.loc[i,'GPT3_response']=completion.choices[0].text


    df.to_csv("progress_singular_gpt3.csv",index=False)
if __name__ == '__main__':
    main()
