import pandas as pd
from tqdm import tqdm
import openai
import re
import os
import csv
from pathlib import Path


#Paste your API key here
INT='CS'
AA = ""
#input your initials here:
TASK = 'one_shot'
# please change the path prompts
PATH = '/Users/shawnchen/Downloads/dev_dataset/'

health_01 = "Context: {context} \n Question: Is this (0) no advice, (1) strong advice, or (2) weak advice?"
health_02 = "Context: {context} \n Question: Does this claim have (1) strong advice, (2) weak advice statement, or there is (0) no advice?"
#format string for pubmed
pubmed_02 = 'Context: {context} \n Question: Does 1 - correlation, 2 - conditional causation, or 3 – direct causation expressed in the sentence, or it is a 0 - no relationship sentence?'
pubmed_01 = "Context: {context} \n Question: Is this a: 0) None, 1) Correlational, 2) Conditional causal, 3) Direct causal?"

def extract_text_files(file_path):
    # Open the file and read its contents into a string variable
    with open(file_path, "r") as f:
        txt_contents = f.read()

    # Return the list of text files
    return str(txt_contents).split('\n\n')[1:4]

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


def prompt_answers(df, filename='', prompt='', prefix='', k=0):
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
      for label, sentence in tqdm(zip(df['label'][k:], df['sentence'][k:]), total=len(df['sentence'][k:])): 
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
      for label, sentence in tqdm(zip(df['label'], df['sentence']), total=len(['sentence'])): 
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

    # Shan runs the following code
    diss_v2 = extract_text_files(PATH+'4_shots_prompts/advice/discussion-v2yy_4.txt')
    unstructured_v2 = extract_text_files(PATH+'4_shots_prompts/advice/unstructured-v2yy_4.txt')
    structured_v2 = extract_text_files(PATH+'4_shots_prompts/advice/structured-v2yy_4.txt')

    #  yy runs the following code 
    # diss_v1 = extract_text_files(PATH+'4_shots_prompts/advice/discussion-v1yy_4.txt')
    # unstructured_v1 = extract_text_files(PATH+'4_shots_prompts/advice/unstructured-v1yy_4.txt')
    # structured_v1 = extract_text_files(PATH+'4_shots_prompts/advice/structured-v1yy_4.txt')

    # prompt_answers(health_advice_diss, filename=cwd+'health_advice_diss_c1_v2.csv', prompt=health_02, prefix=diss_v2[1])
    # prompt_answers(health_advice_un, filename=cwd+'health_advice_un_c1_v2.csv', prompt=health_02, prefix=unstructured_v2[1])
    prompt_answers(health_advice, filename=cwd+'health_advice_struc_c1_v21.csv', prompt=health_02, prefix=structured_v2[1], k=851) # 851/1196

    # prompt_answers(health_advice_diss, filename=cwd+'health_advice_diss_c2_v2.csv', prompt=health_02, prefix=diss_v2[2])
    # prompt_answers(health_advice_un, filename=cwd+'health_advice_un_c2_v2.csv', prompt=health_02, prefix=unstructured_v2[2])
    prompt_answers(health_advice, filename=cwd+'health_advice_struc_c2_v21.csv', prompt=health_02, prefix=structured_v2[2], k=1070) #1070/1196

    prompt_answers(health_advice_diss, filename=cwd+'health_advice_diss_c0_v21.csv', prompt=health_02, prefix=diss_v2[0], k=523) #523/786
    # prompt_answers(health_advice_un, filename=cwd+'health_advice_un_c0_v2.csv', prompt=health_02, prefix=unstructured_v2[0])
    prompt_answers(health_advice, filename=cwd+'health_advice_struc_c0_v21.csv', prompt=health_02, prefix=structured_v2[0], k=876) #876/1196

    print('##### DONE! #####')


if __name__ == "__main__":
    main()

