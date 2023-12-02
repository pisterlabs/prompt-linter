import pandas as pd
from tqdm import tqdm
import openai
import re
import os
import csv
from pathlib import Path


#Paste your API key here
AA = ""

def extract_text_files(file_path):
    # Open the file and read its contents into a string variable
    with open(file_path, "r") as f:
        txt_contents = f.read()

    # Return the list of text files
    return str(txt_contents)

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
    
    #read in the prompts
    pb = pd.read_csv('../../prompts/cot_prompts-4shots/4cot_pubmed.csv')
    hp = pd.read_csv('../../prompts/cot_prompts-4shots/4cot_cs_db_5.csv') 

    pubyy = pb.yy
    yy = hp.yy
    cs = hp.cs

    #Health advice
    #davincci003:
    prompt_answers(health_advice_diss, OUT+'health_3cot/health_diss_yy_3cot.csv', yy[0])
    prompt_answers(health_advice_un, OUT+'health_3cot/health_unstruc_yy_3cot.csv', cs[1])
    prompt_answers(health_advice, OUT+'health_3cot/health_struc_yy_3cot.csv', cs[2])
    #turbo0301:
    prompt_answers(health_advice_diss, OUT+'health_3cot/health_diss_yy_3cot_tb.csv', yy[0], model='gpt-3.5-turbo')
    prompt_answers(health_advice_un, OUT+'health_3cot/health_unstruc_yy_3cot_tb.csv', cs[1], model='gpt-3.5-turbo')
    prompt_answers(health_advice, OUT+'health_3cot/health_struc_yy_3cot_tb.csv', cs[2], model='gpt-3.5-turbo')

    prompt_answers(pubmed_causal, OUT+'pubmed_3cot/pubmed_yy_3cot_tb.csv', pubyy[0], model='gpt-3.5-turbo')
    prompt_answers(pubmed_causal, OUT+'pubmed_3cot/pubmed_yy_3cot.csv', pubyy[0], model='text-davinci-003')


    print('##### DONE! #####')


if __name__ == "__main__":
    main()

