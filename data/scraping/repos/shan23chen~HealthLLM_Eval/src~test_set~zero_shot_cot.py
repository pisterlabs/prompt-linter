import pandas as pd
from tqdm import tqdm
import openai
import re
import os
import csv
from pathlib import Path

#Paste your API key here
AA = ""
TASK = 'zero_shot_cot'

COT = '\n Let\'s think step by step'
#format string for health gs01
health_01 = 'Context: {context} \n Label the sentence as strong medical advice, weak medical advice or no medical advice' + COT
#format string for pubmed dbp1
pubmed_01 = 'Context: {context} \n Question: What type of relationship is this describing? Select only one from:  0 - no relationship, 1 - correlation, 2 - conditional causation, or 3 – direct causation.' + COT



openai.api_key = AA
def gpt3_response(p_text, model="text-davinci-003", temperature=0, max_tokens=256, top_p=1, frequency_penalty=0, presence_penalty=0, best_of=1):
  if model == "gpt-3.5-turbo":
    response = openai.ChatCompletion.create(
    model=model,
    messages=[
        {"role": "user", "content": p_text},
    ],
    temperature=0,
    )['choices'][0]['message']['content']
    return response
  else:
    return openai.Completion.create(
      model=model,
      prompt=p_text,
      temperature=temperature,
      max_tokens=max_tokens,
      top_p=top_p,
      frequency_penalty=frequency_penalty,
      presence_penalty=presence_penalty
    )['choices'][0]["text"]

def prompt_answers(df, filename='', prompt='', prefix='', fold='', model='text-davinci-003'):
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
      writer.writerow(['label', 'context', 'output', 'fold'])

      # write each row of data
      for label, sentence, fold in tqdm(zip(df['label'], df['sentence'], df['fold']), total=len(df['sentence'])): 
        direct_output = gpt3_response(str(prefix+'\n'+prompt.format(context=sentence)), model) #for Bob, change this function
        temp = [label, sentence, direct_output, fold]
        writer.writerow(temp)

def prompt_answers1(df, filename='', prompt='', prefix='', fold='', model='text-davinci-003'):
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
      writer.writerow(['label', 'context', 'output', 'fold'])

      # write each row of data
      for label, sentence, fold in tqdm(zip(df['label'], df['sentence'], df['fold']), total=len(df['sentence'])): 
        direct_output = gpt3_response(str(prefix+'\n'+prompt.format(context=sentence)), model) #for Bob, change this function
        temp = [label, sentence, direct_output, fold]
        writer.writerow(temp)

def main():
    cwd = os.getcwd()

    #fix this output path
    OUT = '../../test_outputs/'
    # read three datasets
    health_advice_diss = pd.read_csv('../../data/test_data_4folds/health_advice_discuss_annotation_test_folds.csv')
    health_advice_un = pd.read_csv('../../data/test_data_4folds/health_advice_unsturctured_abs_annotation_test_folds.csv')
    health_advice = pd.read_csv('../../data/test_data_4folds/health_advice_sturctured_abs_annotation_test_folds.csv')
    pubmed_causal = pd.read_csv('../../data/test_data_4folds/pubmed_causal_annotation_test_folds.csv')

    #Health advice
    #davincci003:
    prompt_answers(health_advice_diss, OUT+'health_0cot/health_diss_0cot.csv', health_01)
    prompt_answers(health_advice_un, OUT+'health_0cot/health_unstruc_0cot.csv', health_01)
    prompt_answers(health_advice, OUT+'health_0cot/health_struc_0cot.csv', health_01)
    #turbo0301:
    prompt_answers(health_advice_diss, OUT+'health_0cot/health_diss_0cot_tb.csv', health_01, model='gpt-3.5-turbo')
    prompt_answers(health_advice_un, OUT+'health_0cot/health_unstruc_0cot_tb.csv', health_01, model='gpt-3.5-turbo')
    prompt_answers(health_advice, OUT+'health_0cot/health_struc_0cot_tb.csv', health_01, model='gpt-3.5-turbo')

    prompt_answers(pubmed_causal, OUT+'pubmed_3cot/pubmed_0cot_tb.csv', pubmed_01, model='gpt-3.5-turbo')
    prompt_answers(pubmed_causal, OUT+'pubmed_3cot/pubmed_0cot.csv', pubmed_01, model='text-davinci-003')
    print('##### DONE! #####')


if __name__ == "__main__":
    main()