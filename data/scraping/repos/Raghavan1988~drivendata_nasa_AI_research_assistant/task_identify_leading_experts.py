"""
Terminal application to identify collaborators in a field of research.
Identifying the leading experts and potential collaborators in a particular domain or domains
"""
import json 
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
import time
from annoy import AnnoyIndex
import operator
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from getpass import getpass

path1 = "arxiv_12_28_2022.json"
path2 = "since_dec22.json"

data_dict = {}

def preprocess(path):
    start = time.time()
    data = []
    print("Begin Loading of Json file:"+path)
    with open(path,'r') as f:
        for line in f:
            data.append(json.loads(line))
    end = time.time()
    print("Time taken to load json file: ",end-start)

    start = time.time()
    sents = []
    for i in range(len(data)):
        sents.append(data[i]['title']+'[SEP]'+data[i]['abstract'])
    end = time.time()
    print("Time taken to create sentences: ",end-start)  

    data_dict[path] = data

    return sents

def generate_annoy(fn):
    n_trees = 256           #Number of trees used for Annoy. More trees => better recall, worse run-time
    embedding_size = 768    #Size of embeddings
    top_k_hits = 5        #Output k hits

    annoy_index = AnnoyIndex(embedding_size, 'angular')


    annoy_index.load(fn)
    return annoy_index

def search(query,annoy_index,model, topK):
    query_embedding = model.encode(query,convert_to_numpy=True)
    tokens = query.strip().split()
    

    hit_dict = {}
    for i in range(0,len(tokens)):

        gram = ' '.join(tokens[:i])
        print (gram.strip())
        if (len(gram) < 3):
            continue
        hits = annoy_index.get_nns_by_vector(query_embedding, topK, include_distances=True)
        for hit in hits[0]:
            if hit not in hit_dict:
                hit_dict[hit] = 0
            hit_dict[hit] += 1

    sorted_hits = sorted(hit_dict.items(), key=lambda item: item[1], reverse=True)    
    hits = []
    for (x,y) in sorted_hits:
        hits.append(x)
    return [hits, hits]

    
def collect_results(hits,sents, path, prefix=""):
    print ("entering results")
    response = ""
    for i in range(len(hits[0])):
        print ("result " + prefix + str(i))
        response += "result " + prefix + str(i) + "\n"
        response += "id:" + data_dict[path][hits[0][i]]['id'] + "\n"
        response += "Title:" + sents[hits[0][i]].split('[SEP]')[0] + "\n"
        response += "Abstract: " + sents[hits[0][i]].split('[SEP]')[1] + "\n"
        response += "Authors:" +  data_dict[path][hits[0][i]]['authors'] + "\n"
        response += "Published: " + data_dict[path][hits[0][i]]['versions'][0]['created'] + "\n"
        print(sents[hits[0][i]].split('[SEP]')[0])
        print(sents[hits[0][i]].split('[SEP]')[1])
        print ("----------------------------------")
    return response



##### Start code
model = SentenceTransformer('sentence-transformers/allenai-specter', device='cpu')
OPENAI_API_KEY = getpass("Please enter your OPEN AI API KEY to continue:")
## load the OPENAI LLM model
open_ai_key = OPENAI_API_KEY

sents_from_path1 = preprocess(path1)
sents_from_path2 = preprocess(path2)

## indices from Annoy
an = generate_annoy("annoy_index.ann")
an2 = generate_annoy("annoy_index_since_dec22.ann")

## load the OPENAI LLM model
llm = OpenAI(openai_api_key=open_ai_key, model_name= "gpt-3.5-turbo-16k")


while True:
    query = input("query:")
    print (query)
    response1 = collect_results(search(query,an,model,20),sents_from_path1, path1)
    response2 = collect_results(search(query,an2,model,20),sents_from_path2, path2)

    response = response1 + "\n" + response2
    if (len(response) > 70000):
        response = response[:70000]
    print (len(response))


    template = """ Take a deep breath. 
    You are a researcher in the FIELD of {query}. You are tasked with finding the leading authors in this field. User entered the FIELD. Search engine returned the {response}. Analyze the results and follow the instructions below to identify leading authors in the field of {query} using the CRITERIA below. 
    CRITERIA:
    1. Author MUST HAVE published at least 2 papers in the FIELD.
    2. Author MUST HAVE published at least 1 paper in the last 2 years. 
    3. Author has COLLABORATED with at least 1 other author in the field.
    4. Author's work has breadth and depth in the FIELD.
    5. Prefer recent authors.
    OUTPUT
    1. List of authors that meet the CRITERIA above as a JSON ARRAY following TYPESCRIPT SCHEMA
    2. It is a JSON array of authors with name, list of IDs of their articles and summary of their work in the FIELD.
    3. REASON why the author is selected and how their work is relevant in the FIELD in 20 WORDS or lessmat.
    4. Output SHOULD STRICTLY FOLLOW JSON SCHEMA
    SCHEMA
    class Author
        name: string
        id: string[id1, id2]
        summary: string
    """
    prompt = PromptTemplate(template=template, input_variables=["query", "response"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    try:
        output = llm_chain.run({'query': query, 'response':response})
        print (output)
    except Exception as e:
        print (e)
