from datetime import datetime
import os
import time
# import chromadb
import numpy as np
# from chromadb.utils import embedding_functions
import pickle
import openai
from sklearn.decomposition import PCA
from tqdm import tqdm
import cohere
import voyageai
from voyageai import get_embeddings,get_embedding
import requests
import json
from transformers import AutoModel
import string
import tiktoken
from dotenv import load_dotenv
load_dotenv()
from numpy.linalg import norm
# history=[]
# history.append({"role": "system", "content": client.starting_prompt})
# history.append({"role": "user", "content": message})


def rfind_punctuation(s, start, end):
    for i in range(end-1, start-1, -1):  # end-1 because Python slices are exclusive at the end
        if s[i] in string.punctuation:
            return i
    return -1  # If no punctuation is found

def token_size(sentence):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(sentence))
def send_split_message_user(response, token_limit=300):
    msg_list = []
    # print(token_limit)
    tokens = token_size(response)

    if tokens > token_limit:
        start = 0
        while start < len(response):
            end = start
            while end < len(response) and token_size(response[start:end]) < token_limit:
                end += 1

            if end < len(response):
                # Look for a suitable split position
                split_pos = response.rfind('\n\n', start, end)
                if split_pos == -1:
                    split_pos = response.rfind('\n', start, end)
                if split_pos == -1:
                    split_pos = rfind_punctuation(response, start, end)
                if split_pos == -1:
                    split_pos = response.rfind(' ', start, end)
                if split_pos == -1 or split_pos <= start:
                    split_pos = end - 1

                msg_list.append(response[start:split_pos].strip())
                start = split_pos + 1
            else:
                # Add the last chunk
                msg_list.append(response[start:end].strip())
                break
    else:
        msg_list.append(response)

    return msg_list

def wizard_coder(history: list[dict]):
    DEFAULT_SYSTEM_PROMPT = history[0]['content']+'\n\n'
    B_INST, E_INST = "### Instruction:\n", "\n\n### Response:\n"
    messages = history.copy()
    messages_list=[DEFAULT_SYSTEM_PROMPT]
    messages_list.extend([
        f"{B_INST}{(prompt['content']).strip()}{E_INST}{(answer['content']).strip()}\n\n"
        for prompt, answer in zip(messages[1::2], messages[2::2])
    ])
    messages_list.append(f"{B_INST}{(messages[-1]['content']).strip()}{E_INST}")
    return "".join(messages_list)

def gpt(history: list[dict]):
    l=[x['content'] for x in history]
    return '\n---\n'.join(l)


def string_subtraction(main_string, sub_string):
    return main_string.replace(sub_string, '', 1)  # The '1' ensures only the first occurrence is removed

'''
Traverse through files
'''
def traverse_files(path, file_format, start_folder_name):
    # Ensure valid file format
    if file_format not in ['rst', 'md']:
        raise ValueError("Invalid file format. Allowed formats: 'rst', 'md'")

    results = []

    # Check if the provided path exists
    if not os.path.exists(path):
        raise ValueError(f"The provided path '{path}' does not exist.")
    # print(os.walk(path))
    folder_tree = f"{start_folder_name} (h1)\n"
    for root, dir, files in os.walk(path):
        # print(root, dir, files)

        for file in files:
            if file.endswith('.pkl'):
                path_list = [start_folder_name] + string_subtraction(root, path).split('/')[1:]
                line = ((len(path_list)-1)*"--" + path_list[-1] + f" (L{len(path_list)})")
                folder_tree += f"{line}\n"
    # print(tree)

    for root, dir ,files in os.walk(path):
        # print(root, dir, files)
        for file in files:
            if file.endswith('.pkl'):
                # file path
                file_path = os.path.join(root, file)
                path_list = [start_folder_name] + string_subtraction(root, path).split('/')[1:]
                with open(file_path, 'rb') as pkl_file:
                    content = pickle.load(pkl_file)
                # print(path_list)
                folder_path = ' > '.join(f"{item} (Level{i+1})" for i, item in enumerate(path_list))
                # print(content)
                results.append(([folder_tree, folder_path], content))
    return results


start=time.time()
# Process each page
# docs = traverse_files("/home/bot/dataset/edugpt/Scrape_header/ROS", "md", "ROS")
docs = traverse_files("/home/bot/dataset/edugpt/Scrape_rst/Sawyer", "rst", "Sawyer")
for i in docs:
    print(i)
# human_embedding_prompt need to include toc_link, toc_page, title, section
# mention the format required
# system_embedding_prompt= 'you need to read a research section of paper and try to paraphrase the content with description, related tasks and scenario it can help.'
# TODO TECHNIQUE
# technique = 'none'
# technique = 'bullet'
# technique = 'connected_bullet'
# technique = 'seperate_paragraph'
# technique = 'seperate_sentence'
technique = 'recursive_seperate'
# TODO METHOD
# method='to_task'
# method='to_doc'
# method='to_doc_chat_completion'
# method = 'to_task_chat_completion'
method='none'
# method='sum'fail = []

# TODO MODEL
# model='local'
model='openai'
# model='cohere'
# model='jina'
# model='zephyr'
# EDUGPT
# 5ZqmnmIRFFA6u4sgnbcDyEDrHefTeTqAWTKi0e34
# model='voyage'

system_embedding_prompt = ''
system_query_prompt = ''


if method=='to_task':
    system_embedding_prompt = ("Given the content and the document_hierarchy_path of a document, describe the tasks you can answer based on its content.")
    system_query_prompt = 'Rephrase the provided task in your own words without changing its original meaning.'
elif method=='to_doc':
    system_embedding_prompt = ("Summarize the content of the given document and its document_hierarchy_path. Think of the related tasks and scenarios it can help with.")
    system_query_prompt = 'Given the task, generate a document that can help you to answer this task.'
elif method=='sum':
    system_embedding_prompt = "Summarize"


'''
bullet points
'''
def chat_completion(system_message, human_message):
    print("start_chat_completion")
    system_message = system_message
    messages=[{"role": "system", "content": system_message}, {"role": "user", "content": human_message}]
    # if model=='local':
    #     prompt=wizard_coder(history)
    # elif model=='openai':
    #     prompt=gpt(history)
    # print(prompt)
    completion = openai.ChatCompletion.create(
        model='gpt-3.5-turbo', messages=messages, temperature=0
    )
    # print(completion)

    answer=completion['choices'][0]['message']["content"]
    # print(answer)
    print("end_chat_completion")
    print("----------------------")
    return answer

# folder_tree, folder_path, segment_tree, segment_path, segment
# TODO path vs level
# TODO summary vs description
human_embedding_prompt= 'document_hierarchy_path: {segment_path}\ndocument: {segment}\n'
system_query_prompt= 'Rephrase the provided task in your own words without changing its original meaning.'
# system_query_prompt= 'Given a primary task, please list and describe the associated tasks'

'''
# This part is embedding
'''
print('read time:',time.time()-start)
print(len(docs))
start=time.time()


for n in [900,800,700,600,500,400,300,200,100]:
    print(n)
    if model=='local' or model=='zephyr':
        openai.api_key = "empty"
        openai.api_base = "http://localhost:8000/v1"
    elif model=='openai':
        openai.api_key = os.getenv("OPENAI_API_KEY")
    elif model=='cohere':
        co = cohere.Client(os.getenv("COHERE_API_KEY"))
    elif model=='voyage':
        voyageai.api_key = os.getenv("VOYAGE_API_KEY")
    elif model=='jina':
        jina = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
    fail = []


    id_list = []
    doc_list=[]
    embedding_list=[]
    # Define the path to your file
    # file_path = 'documents/EECS106A_textbook.txt'

    # Open the file and read its contents into a string
    # with open(file_path, 'r') as file:
    #     segment = file.read()

    # Now file_contents holds the contents of the file as a string

    # docs.append((["textbook", "eecs"], [{'Page_table': "page_toc", 'Page_path': "page_path", 'Segment_print': segment}]))
    for doc in tqdm(docs, desc="Processing documents"):
        document = []
        ids = []
        embedding = []
        folder = doc[0]
        file = doc[1]
        # print(doc)
        # elements needed for embedding
        folder_tree = folder[0]
        folder_path = folder[1]
        if not file:
            continue
        for chunk in file:
            # elements needed for embedding
            segment_tree = chunk['Page_table']
            segment_path = chunk['Page_path'].split('\n')[-1]
            segment = chunk['Segment_print']
            sp = system_embedding_prompt
            count = 1
            if technique == 'seperate_paragraph':
                segment = [part for part in segment.split('\n\n') if part]
            elif technique == 'seperate_sentence':
                segment = [part for part in segment.split('\n') if part]
            elif technique == 'connected_bullet':
                system_message='Your task is to summarize passage given by the user into bullet points'
                segment = [chat_completion(system_message, segment)]
            elif technique == 'bullet':
                system_message = 'Your task is to summarize passage given by the user into bullet points'
                segment = chat_completion(system_message, segment)
                segment = segment.strip().split('\n')

                # Removing the '- ' prefix from each sentence
                segment = [constraint[2:] if constraint.startswith('- ') else constraint for constraint in segment]
            elif method == 'to_task_chat_completion':
                system_message = 'Given the content and the document_hierarchy_path of a document, describe the questions you can ask based on its content.'
                segment = chat_completion(system_message, segment)
                segment = segment.strip().split('\n')
                segment = [line for line in segment if line and line[0].isdigit()]
            elif technique == 'recursive_seperate':
                n=n
                # print(segment)
                segment = send_split_message_user(segment, n)
                if False:
                    for i,seg in enumerate(segment):
                        print(i,[seg])
                    print("---")
            elif technique == 'none':
                segment = [segment]
            for smaller_chunk in segment:
                # print(smaller_chunk)
                hp = human_embedding_prompt.format(segment=smaller_chunk, segment_path=folder_path + " > " + segment_path)
            # history = [{"role": "system", "content": sp}, {"role": "user", "content": hp}]
            # input = wizard_coder(history)
            # print(segment)
            # embedding.append(openai.Embedding.create(model="text-embedding-ada-002", input=input)['data'][0]['embedding'])
            # ids.append(segment_path)
            # document.append(segment)
            #     print(smaller_chunk)
            #     print(smaller_chunk)
                try:
                    if method == 'none' or method =='to_doc_chat_completion':
                        history = [{"role": "user", "content": hp.strip()}]
                    else:
                        history = [{"role": "system", "content": sp}, {"role": "user", "content": hp}]
                    if model == 'local':
                        # print(input)
                        # embedding.append(openai.Embedding.create(model="text-embedding-ada-002", input=wizard_coder(history))['data'][0]['embedding'])
                        embedding.append(openai.Embedding.create(model="text-embedding-ada-002", input=hp.strip())['data'][0]['embedding'])
                    elif model == 'zephyr':
                        embedding.append(openai.Embedding.create(model="text-embedding-ada-002", input=gpt(history))['data'][0]['embedding'])
                    elif model == 'openai':
                        embedding.append(openai.Embedding.create(model="text-embedding-ada-002", input=gpt(history))['data'][0]['embedding'])
                    elif model == 'cohere':
                        embedding.extend(co.embed(texts=[hp],
                                         model="embed-english-v3.0",
                                         input_type="search_document").embeddings)
                    elif model == 'voyage':
                        time.sleep(1)
                        embedding.append(get_embedding(hp, model="voyage-01"))
                    elif model == 'jina':
                        embedding.append(jina.encode([hp])[0])
                    id = folder_path + " > " + segment_path + f"({count})"
                    ids.append(id)
                    # print(type(smaller_chunk), len(smaller_chunk))
                    doc_list.append(smaller_chunk)

                except openai.error.APIError as e:
                    print(f"Embedding error: {e}")
                    fail.append(folder_path + " > " + segment_path)
                count += 1

        # for id in ids:
        #     print(id)
        # print("----------------------")
        id_list.extend(ids)
        # doc_list.extend(document)
        embedding_list.extend(embedding)
        # collection.add(documents=document, ids=ids, embeddings=embedding)
    id_list=np.array(id_list)
    doc_list=np.array(doc_list)
    embedding_list=np.array(embedding_list)
    print(id_list.shape)
    print(doc_list.shape)
    print(embedding_list.shape)
    print('create time:',time.time()-start)

    # Store the variables in a dictionary
    data_to_store = {
        'id_list': id_list,
        'doc_list': doc_list,
        'embedding_list': embedding_list
    }

    # Define the folder name
    folder_name = "pickle"

    # Create the folder if it does not exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Change the current working directory to the new folder
    os.chdir(folder_name)

    print("Current Working Directory:", os.getcwd())

    # Open a file in binary write mode and store the data using pickle
    if technique=='recursive_seperate':
        with open(f'{technique}_{method}_{model}_embedding_{n}.pkl', 'wb') as f:
            pickle.dump(data_to_store, f)
    else:
        with open(f'{technique}_{method}_{model}_embedding.pkl', 'wb') as f:
            pickle.dump(data_to_store, f)

    print("failed embeddings")
    os.chdir("..")
    for i in fail:
        print(i)