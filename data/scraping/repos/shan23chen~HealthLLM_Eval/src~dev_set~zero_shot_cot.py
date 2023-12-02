import pandas as pd
from tqdm import tqdm
import openai
import re
import os
import csv
from pathlib import Path

# Note just need change AA, INT, and the four format strings

#Paste your API key here
AA = ""
#input your initials here:
INT = 'db'
TASK = 'zero_shot_cot'

COT = '\n Let\'s think step by step'
#format string for health
health_01 = "Context: {context} \n Question: Is this (0) no advice, (1) strong advice, or (2) weak advice?" + COT
health_02 = "Context: {context} \n Question: What type of advice is this? Select only one from: 0 – no advice, 1 - strong advice, or 2 - weak advice." + COT
#format string for pubmed
pubmed_01 = 'Context: {context} \n Question: What type of relationship is this describing? Select only one from:  0 - no relationship, 2 - correlation, 3 - conditional causation, or 4 – direct causation.' + COT
pubmed_02 = "Context: {context} \n Question: Is this describing a (1) directly correlative relationship, (2) conditionally causative relationship, (3) causative relationship, or (0) no relationship." + COT



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


def prompt_answers(df, filename='', prompt=''):
    '''
    df: dataset/dataframe - you want to pass in: health_advice, press_release, pubmed_causal
    prompt: how do you want to prompt the input
    '''
    print(f'**** writing to {filename} for zero_shot on health device:')
    # open a CSV file for writing
    with open(filename, 'w', newline='') as file:
      # create a CSV writer object
      writer = csv.writer(file)

      # write the header row
      writer.writerow(['type', 'label', 'context', 'output'])

      # write each row of data
      for type, label, sentence in tqdm(zip(df['type'], df['label'], df['sentence']), total=len(df['sentence'])): 
        direct_output = gpt3_response(prompt.format(context=sentence)) # for Bob, change this sentence
        temp = [type, label, sentence, direct_output]
        writer.writerow(temp)

def prompt_answers1(df, filename='', prompt=''):
    '''
    df: dataset/dataframe - you want to pass in: health_advice, press_release, pubmed_causal
    prompt: how do you want to prompt the input
    '''
    print(f'#### writing to {filename} for zero_shot on pubmed causal:')
    # open a CSV file for writing
    with open(filename, 'w', newline='') as file:
      # create a CSV writer object
      writer = csv.writer(file)

      # write the header row
      writer.writerow(['label', 'context', 'output'])

      # write each row of data
      for label, sentence in tqdm(zip(df['label'], df['sentence']), total=len(df['sentence'])): 
        direct_output = gpt3_response(prompt.format(context=sentence)) #for Bob, change this function
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
    press_release = pd.read_csv('press_release_causal_language_annotation.csv')
    pubmed_causal = pd.read_csv('pubmed_causal_language_annotation.csv')

    # diss: disscussion, un: unstrcutured, struc: structured
    health_advice_diss['type'] = ['diss' for i in health_advice_diss.label]
    health_advice_un['type'] = ['un' for i in health_advice_un.label]
    health_advice['type'] = ['struc' for i in health_advice.label]
    health = pd.concat([health_advice_diss, health_advice_un, health_advice]) 

    # prompt_answers(health, cwd+'health_01.csv', health_01)
    # prompt_answers(health, cwd+'health_02.csv', health_02)
    prompt_answers1(pubmed_causal, cwd+'pubmed_01.csv', pubmed_01)
    prompt_answers1(pubmed_causal, cwd+'pubmed_02.csv', pubmed_02)
    print('##### DONE! #####')


if __name__ == "__main__":
    main()