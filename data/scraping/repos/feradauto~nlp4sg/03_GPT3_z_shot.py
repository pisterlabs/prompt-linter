import os
import sys
import openai
import pandas as pd
import numpy as np
import argparse
import torch
import operator
from sklearn.metrics import classification_report
openai.api_key = os.getenv("OPENAI_API_KEY")

def main():
    data_path="../../data/"
    outputs_path="../../outputs/"

    df_test_final=pd.read_csv(outputs_path+"general/test_set_final.csv")
    df_test_final=df_test_final.iloc[:2000].reset_index(drop=True)
    df=df_test_final.reset_index(drop=True).copy()

    preprompt="There is an NLP paper with the following title:\n"

    #question="Could it be argued that this paper directly contributes to the UN Sustainable Development Goals? Answer yes or no."
    #question="Is this paper directly or indirectly contributing to the UN Sustainable Development Goals? Answer yes or no, then explain."
    question="Is this paper contributing to the UN Sustainable Development Goals? Answer yes or no. If the answer is \"yes\", mention which goal the paper is contributing to and in which way."
    df=df.assign(statement=preprompt+df.title_clean+"\n"+question)
    for i,d in df.iterrows():
        input_prompt=d['statement']
        completion = openai.Completion.create(engine="text-davinci-002", prompt=input_prompt,temperature=0,max_tokens=50,logprobs=10)
        dict_norm={}
        dict_uniques={}
        ## new part, get the first element where there the most probable is not a break of line
        for choices in completion['choices'][0]["logprobs"]['top_logprobs']:
            choice=dict(choices)
            max_val=max(choice.items(), key=operator.itemgetter(1))[0]
            if "\n" not in max_val:
                elements=choices
                break
        for e in elements:
            e_modified=e.lower().lstrip(' ')
            if e_modified in dict_uniques:
                dict_uniques[e_modified]=dict_uniques[e_modified]+np.exp(elements[e])
            else:
                dict_uniques[e_modified]=np.exp(elements[e])

        if ('no' in dict_uniques.keys()) and ('yes' in dict_uniques.keys()):
            dict_norm={'no':dict_uniques['no'],'yes':dict_uniques['yes']}
        elif ('no' in dict_uniques.keys()):
            dict_norm={'no':dict_uniques['no'],'yes':0}
        elif ('yes' in dict_uniques.keys()):
            dict_norm={'no':0,'yes':dict_uniques['yes']}

        factor=1.0/sum(dict_norm.values())
        for k in dict_norm:
            dict_norm[k] = dict_norm[k]*factor    

        df.loc[i,'full_prompt']=input_prompt
        df.loc[i,'GPT3_response']=completion.choices[0].text
        df.loc[i,'proba_1']=dict_norm['yes']
        df.loc[i,'proba_0']=dict_norm['no']


    df=df.assign(prediction_proba=np.where(df.proba_1>0.5,1,0))

    df=df.assign(textual_pred=np.where(((df.GPT3_response.str.lower().str.startswith("yes")) | 
                                         (df.GPT3_response.str.lower().str.contains("is contrib"))
                                         ),1,0))

    df.to_csv(outputs_path+"sg_classifier/gpt3_zshot_title_f.csv",index=False)
    
if __name__ == '__main__':
    main()
