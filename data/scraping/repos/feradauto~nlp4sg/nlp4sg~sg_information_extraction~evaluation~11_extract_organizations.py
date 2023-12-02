import time
import os
import pandas as pd
import numpy as np
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")


def request_gpt3(positives):
    ## format inputs
    positives=positives.assign(abstract_for_prompt=positives.abstract_for_prompt.fillna(""))
    positives=positives.assign(acknowledgments_for_prompt=positives.acknowledgments_for_prompt.fillna(""))

    positives=positives.assign(paper_text="Title: "+positives.title_clean+"\n"+positives.abstract_for_prompt+
                              positives.acknowledgments_for_prompt)

    ack_preprompt="""Identify the organizations mentioned in the following paper. Respond with the organizations separated by commas. Answer "No organizations" if there aren't any organizations in the text:"""

    ack_postprompt="""Which are the organizations mentioned?"""

    positives=positives.assign(org_prompt_text=ack_preprompt+"\n"+positives.paper_text+"\n"+ack_postprompt)
    positives=positives.loc[~positives.acknowledgments_clean.isna()].reset_index(drop=True)
    # request
    for i,d in positives.iterrows():
        input_prompt=d['org_prompt_text']
        #completion = openai.Completion.create(engine="text-davinci-002", prompt=input_prompt,temperature=0,max_tokens=40)
        positives.loc[i,'GPT3_response']=" "#completion.choices[0].text

    
    positives=positives.assign(clean_response=positives.GPT3_response.replace("\n","",regex=True))
    positives=positives.assign(clean_response=positives.clean_response.str.split(","))
    
    orgs=positives.loc[:,['ID','clean_response']]
    orgs=orgs.explode("clean_response")
    orgs=orgs.rename(columns={'clean_response':'organization'})
    orgs=orgs.assign(organization=orgs.organization.str.lstrip().str.rstrip())
    return positives,orgs



def main():
    data_path="../../data/"
    output_path="../../outputs/"
    positives=pd.read_csv(output_path+"sg_ie/positives_ready.csv")
    
    positives,orgs=request_gpt3(positives)
    
    positives.to_csv(output_path+"sg_ie/GPT3_responses_org.csv",index=False)
    orgs.to_csv(output_path+"sg_ie/organizations_GPT3.csv",index=False)
    
    
if __name__ == '__main__':
    main()