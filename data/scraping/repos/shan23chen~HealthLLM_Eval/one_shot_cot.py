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
TASK = 'one_shot_cot'
# please change the path prompts
PATH = '/Users/shawnchen/Downloads/dev_dataset/4_shots_prompts/cot_prompts-4shots/'

#yy for disscusiion only
health_01 = "Context: {context} \n QUESTION: Does this claim have (2) strong advice, (1) weak advice, or there is (0) no advice? \n Let’s think step by step: "
#bob's for not discussion
health_02 = "Context: {context} \n QUESTION: is this a 2) strong advice, 1) weak advice 0) no advice? \n Let’s think step by step: "
#format string for pubmed db01, yy02
pubmed_01 = 'CONTEXT: {context} \n QUESTION: Is this a: 0) None, 1) Correlational, 2) Conditional causal, 3) Direct causal? \n Let’s think step by step:' 
pubmed_02 = "CONTEXT: {context} \n QUESTION: Does 2 - correlation, 3 - conditional causation, or 4 – direct causation expressed in the sentence, or it is a 1 - no relationship sentence? \n Let’s think step by step: "

def extract_text_files(file_path):
    # Open the file and read its contents into a string variable
    with open(file_path, "r") as f:
        txt_contents = f.read()

    # Return the list of text files
    return str(txt_contents).split('\n\n')

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
    diss_v2 = extract_text_files('/Users/shawnchen/Downloads/dev_dataset/4_shots_prompts/cot_prompts-4shots/discussion-cot_yy_3.txt')[:3]
    unstructured_v2 = extract_text_files('/Users/shawnchen/Downloads/dev_dataset/4_shots_prompts/cot_prompts-4shots/unstructured-cot_shan_3.txt')[:3]
    structured_v2 = extract_text_files('/Users/shawnchen/Downloads/dev_dataset/4_shots_prompts/cot_prompts-4shots/structured-cot_shan_3.txt')[:3]
    pubmedv1 = extract_text_files('/Users/shawnchen/Downloads/dev_dataset/4_shots_prompts/cot_prompts-4shots/pubmed-cot-db.txt')[:4]
    pubmedv2 = extract_text_files('/Users/shawnchen/Downloads/dev_dataset/4_shots_prompts/cot_prompts-4shots/pubmed-cot-yy.txt')[:4]

    # prompt_answers(health_advice_diss, filename=cwd+'health_advice_diss_c1_v2.csv', prompt=health_02, prefix=diss_v2[1])
    # prompt_answers(health_advice_un, filename=cwd+'health_advice_un_c1_v2.csv', prompt=health_02, prefix=unstructured_v2[1])
    # prompt_answers(health_advice, filename=cwd+'health_advice_struc_c1_v2.csv', prompt=health_02, prefix=structured_v2[1])

    # prompt_answers(health_advice_diss, filename=cwd+'health_advice_diss_c2_v2.csv', prompt=health_02, prefix=diss_v2[2])
    # prompt_answers(health_advice_un, filename=cwd+'health_advice_un_c2_v2.csv', prompt=health_02, prefix=unstructured_v2[2])
    # prompt_answers(health_advice, filename=cwd+'health_advice_struc_c2_v2.csv', prompt=health_02, prefix=structured_v2[2])

    # prompt_answers(health_advice_diss, filename=cwd+'health_advice_diss_c0_v2.csv', prompt=health_02, prefix=diss_v2[0])
    # prompt_answers(health_advice_un, filename=cwd+'health_advice_un_c0_v2.csv', prompt=health_02, prefix=unstructured_v2[0])
    # prompt_answers(health_advice, filename=cwd+'health_advice_struc_c0_v2.csv', prompt=health_02, prefix=structured_v2[0])

    # prompt_answers(pubmed_causal, filename=cwd+'pudmed_c0_v1.csv', prompt=pubmed_01, prefix=pubmedv1[0])
    # prompt_answers(pubmed_causal, filename=cwd+'pudmed_c1_v1.csv', prompt=pubmed_01, prefix=pubmedv1[1])
    # prompt_answers(pubmed_causal, filename=cwd+'pudmed_c2_v1.csv', prompt=pubmed_01, prefix=pubmedv1[2])
    # prompt_answers(pubmed_causal, filename=cwd+'pudmed_c3_v1.csv', prompt=pubmed_01, prefix=pubmedv1[3])

    # prompt_answers(pubmed_causal, filename=cwd+'pudmed_c0_v2.csv', prompt=pubmed_02, prefix=pubmedv2[0])
    # prompt_answers(pubmed_causal, filename=cwd+'pudmed_c1_v2.csv', prompt=pubmed_02, prefix=pubmedv2[1])
    prompt_answers(pubmed_causal, filename=cwd+'pudmed_c2_v21.csv', prompt=pubmed_02, prefix=pubmedv2[2],k=348)
    prompt_answers(pubmed_causal, filename=cwd+'pudmed_c3_v2.csv', prompt=pubmed_02, prefix=pubmedv2[3])
    print('##### DONE! #####')


if __name__ == "__main__":
    main()

