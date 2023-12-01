import os, shutil, sys, time, subprocess, json
from decouple import config

from bs4 import BeautifulSoup
# from imageai.Classification import ImageClassification

import openai

execution_path = os.getcwd()
openai.api_key = config('OPENAI_API_KEY')

def scrub_folders():
    for root, dirs, files in os.walk("BancoQuestoes/"):
        if not 'All' in root:     # do not walk over this folder
            for f in files:
                if '.Rnw' in f or '.jpg' in f:
                    shutil.copyfile(f'{root}/{f}', f'BancoQuestoes/All/{f}')
                    tags_full = root.strip('BancoQuestoes/')
                    if '\\' in tags_full:
                        tags_split = tags_full.split('\\')
                        tag1 = tags_split[0]
                        tag2 = ''.join(tags_split[1:])  # descobrir pq come a ultima letra (?)
                        # print(f'{tag1} | {tag2} \t {root}/{f}') 
                    else:
                        tag1 = tags_full
                        tag2 = ""
                    if '.Rnw' in f:
                        add_tags(f'BancoQuestoes/All/{f}', f'#TAGS - TEMA:{tag1} \n#TAGS - SUBTEMA:{tag2} \n')

def add_tags(file_name, text):
    with open(file_name, 'r', encoding='utf-8') as original: 
        data = original.read()
    with open(file_name, 'w', encoding='utf-8') as modified:
        modified.write(text + '\n' + data)
    

def delete_files_Geral():
    files = os.listdir('BancoQuestoes/All/')
    for f in files:
        if '.jpg' in f or '.Rnw' in f or '.html' in f:
            os.remove(f'BancoQuestoes/All/{f}')

# def image_label():
#     files = os.listdir('BancoQuestoes/All/')
#     prediction = ImageClassification()
#     prediction.setModelTypeAsInceptionV3()
#     prediction.setModelPath(os.path.join(execution_path, "inception_v3_weights_tf_dim_ordering_tf_kernels.h5"))
#     prediction.loadModel()
#     for f in files:
#         if '.jpg' in f:
#             predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, f'BancoQuestoes/All/{f}'), result_count=5 )
#             for eachPrediction, eachProbability in zip(predictions, probabilities):
#                 if eachProbability > 0.4:
#                     print(f'Image: {f} \b {eachPrediction}')
#                 else:
#                     break
#             print('\n')    

def text_from_html():
    text_dict = {}
    files = os.listdir('BancoQuestoes/All/')
    for f in files:
        if '.html' in f:
            with open(f'BancoQuestoes/All/{f}', encoding="utf-8") as html_file:
                soup = BeautifulSoup(html_file, 'html.parser')
                text_dict[f] = soup.find('h4').next_sibling.strip('\n')
                # print(soup.find('h4').next_sibling.strip('\n'))
    with open('text_dict.json', 'w', encoding="utf-8") as fp:
        json.dump(text_dict, fp)
    return text_dict


def gpt3_getKeys(text):
    response = openai.Completion.create(
        engine="davinci",
        prompt=f'Text: {text} \nKeywords:',
        temperature=0.3,
        max_tokens=60,
        top_p=1.0,
        frequency_penalty=0.8,
        presence_penalty=0.0,
        stop=["\n"]
    )
    print(text)
    print(response["choices"][0]["text"])
    print(30*'-')
    print('\n\n')
    return response

def gpt3_convert_list(text_dict, n):
    i = 0
    gpt_keywords = {}
    for file_name in text_dict.keys():
        text = text_dict[file_name]
        if len(text) > n:
            keywords = gpt3_getKeys(text)
            gpt_keywords[file_name] = keywords["choices"][0]["text"]
            i += 1
        if i > 50:
            break   
    with open('gpt3_dict.json', 'w', encoding="utf-8") as fp:
        json.dump(gpt_keywords, fp)   

def create_html():
    files = os.listdir('BancoQuestoes/All/')
    for f in files:
        if '.Rnw' in f:
            command = f'Rscript.exe make_html.R {f} {42}'
            subprocess.call(command, cwd="BancoQuestoes/All/")
            # ret = subprocess.run(command, capture_output=True, shell=True, cwd="BancoQuestoes/All")  
            # print(f, ret)

if __name__ == '__main__':
    # delete_files_Geral()
    # scrub_folders()
    # create_html()
    # image_label()
    print(text_from_html())
    pass