import pandas as pd
from tqdm import tqdm
import openai
import re
import os
import csv
from pathlib import Path


#Paste your API key here
INT='CS_db'
AA = ""
#input your initials here:
TASK = '4_shots_cot'
health_01 = "Context: {context} \n Question: Is this (0) no advice, (1) strong advice, or (2) weak advice?"
health_02 = "Context: {context} \n Question: Does this claim have (1) strong advice, (2) weak advice statement, or there is (0) no advice?"
#format string for pubmed
pubmed_02 = 'Context: {context} \n Question: Does 1 - correlation, 2 - conditional causation, or 3 – direct causation expressed in the sentence, or it is a 0 - no relationship sentence?'
pubmed_01 = "Context: {context} \n Question: Is this a: 0) None, 1) Correlational, 2) Conditional causal, 3) Direct causal?" 

holder = ''

def extract_text_files(file_path):
    # Open the file and read its contents into a string variable
    with open(file_path, "r") as f:
        txt_contents = f.read()

    # Return the list of text files
    return str(txt_contents)

openai.api_key = AA
def gpt3_response(p_text, model="text-davinci-003", temperature=0, max_tokens=256, top_p=1, frequency_penalty=0, presence_penalty=0, best_of=1):
  return openai.Completion.create(
    model=model,
    prompt=p_text,
    temperature=temperature,
    max_tokens=max_tokens,
    top_p=top_p,
    frequency_penalty=frequency_penalty,
    presence_penalty=presence_penalty
  )['choices'][0]["text"]


def prompt_answers(df, filename='', prompt='', prefix=''):
    '''
    df: dataset/dataframe - you want to pass in: health_advice, press_release, pubmed_causal
    prompt: how do you want to prompt the input
    '''
    print(f'#### writing to {filename} for four_shots on health advice:')
    # open a CSV file for writing
    with open(filename, 'w', newline='') as file:
      # create a CSV writer object
      writer = csv.writer(file)

      # write the header row
      writer.writerow(['label', 'context', 'output'])

      # write each row of data
      for label, sentence in tqdm(zip(df['label'], df['sentence']), total=len(df['sentence'])): 
        # print(str(prefix+'\n\n'+prompt.format(context=sentence)))
        direct_output = gpt3_response(str(prefix+'\n'+prompt.format(context=sentence))) #for Bob, change this function
        temp = [label, sentence, direct_output]
        writer.writerow(temp)

def prompt_answers1(df, filename='', prompt='', prefix=''):
    '''
    df: dataset/dataframe - you want to pass in: health_advice, press_release, pubmed_causal
    prompt: how do you want to prompt the input
    '''
    print(f'#### writing to {filename} for four_shots on pubmed causal:')
    # open a CSV file for writing
    with open(filename, 'w', newline='') as file:
      # create a CSV writer object
      writer = csv.writer(file)

      # write the header row
      writer.writerow(['label', 'context', 'output'])

      # write each row of data
      for label, sentence in tqdm(zip(df[542:]['label'], df[542:]['sentence']), total=len(df[542:]['sentence'])): 
        direct_output = gpt3_response(str(prefix+'\n'+prompt.format(context=sentence))) #for Bob, change this function
        temp = [label, sentence, direct_output]
        writer.writerow(temp)

def main():
    cwd = os.getcwd()
    cwd = cwd+'/outputs/'+INT+'_'+TASK+'/'
    print(f'outputing into this directory:{cwd}')
    Path(cwd).mkdir(parents=True, exist_ok=True)
    # read three datasets
    health_advice_diss = pd.read_csv('advice_discussion_annotation.csv')
    health_advice_un = pd.read_csv('advice_unstructured_abs_annotation.csv')
    health_advice = pd.read_csv('advice_structured_abs_annotation.csv')
    # press_release = pd.read_csv('press_release_causal_language_annotation.csv')
    pubmed_causal = pd.read_csv('pubmed_causal_language_annotation.csv')
    # ddf = pd.read_csv('/Users/shawnchen/Desktop/4cot_cs_db_5.csv')
    ddf = pd.read_csv('/Users/shawnchen/Desktop/4cot_pubmed.csv') 
    cs = ddf.cs
    db = ddf.db
    hv = ddf.hv
    yy = ddf.yy
    gs = ddf.gs
    # please change the path prompts
    PATH = '/Users/shawnchen/Desktop/advice-v1yy&v2yy_3/'
    # print(extract_text_files(PATH+'4_shots_prompts/advice/discussion-v1_4.txt'))

    # prompt_answers(health_advice_diss, cwd+'health_diss_cs_3cot.csv', cs[0])
    # prompt_answers(health_advice_diss, cwd+'health_diss_db_3cot.csv', db[0])
    # prompt_answers(health_advice_diss, cwd+'health_diss_hv_3cot.csv', hv[0])
    # prompt_answers(health_advice_diss, cwd+'health_diss_yy_3cot.csv', yy[0])
    # prompt_answers(health_advice_diss, cwd+'health_diss_gs_3cot.csv', gs[0])

    # prompt_answers(health_advice_un, cwd+'health_unstruc_cs_3cot.csv', cs[1])
    # prompt_answers(health_advice_un, cwd+'health_unstruc_db_3cot.csv', db[1])
    # prompt_answers(health_advice_un, cwd+'health_unstruc_hv_3cot.csv', hv[1])
    # prompt_answers(health_advice_un, cwd+'health_unstruc_yy_3cot.csv', yy[1])
    # prompt_answers(health_advice_un, cwd+'health_unstruc_gs_3cot.csv', gs[1])

    # prompt_answers(health_advice, cwd+'health_struc_cs_3cot.csv', cs[2])
    # prompt_answers(health_advice, cwd+'health_struc_db_3cot.csv', db[2])
    # prompt_answers(health_advice, cwd+'health_struc_hv_3cot.csv', hv[2])
    # prompt_answers(health_advice, cwd+'health_struc_yy_3cot.csv', yy[2])
    # prompt_answers(health_advice, cwd+'health_struc_gs_3cot.csv', gs[2])

    prompt_answers(pubmed_causal, cwd+'pubmed_cs_3cot.csv', cs[0])
    prompt_answers(pubmed_causal, cwd+'pubmed_db_3cot.csv', db[0])
    prompt_answers(pubmed_causal, cwd+'pubmed_hv_3cot.csv', hv[0])
    prompt_answers(pubmed_causal, cwd+'pubmed_yy_3cot.csv', yy[0])
    prompt_answers(pubmed_causal, cwd+'pubmed_gs_3cot.csv', gs[0])

    # prompt_answers1(pubmed_causal, cwd+'pubmed_v1_4.csv', pubmed_01, prefix=extract_text_files(PATH+'4_shots_prompts/pubmed/v1_4.txt'))
    # prompt_answers1(pubmed_causal, cwd+'pubmed_v2_4.csv', pubmed_02, prefix=extract_text_files(PATH+'4_shots_prompts/pubmed/v1yy_4.txt'))

    # prompt_answers1(pubmed_causal, cwd+'pubmed_v1_4_yy.csv', pubmed_01, prefix=extract_text_files(PATH+'4_shots_prompts/pubmed/v2_4.txt'))
    # prompt_answers1(pubmed_causal, cwd+'pubmed_v2_41_yy.csv', pubmed_02, prefix=extract_text_files(PATH+'4_shots_prompts/pubmed/v2yy_4.txt'))
    print('##### DONE! #####')


if __name__ == "__main__":
    main()

