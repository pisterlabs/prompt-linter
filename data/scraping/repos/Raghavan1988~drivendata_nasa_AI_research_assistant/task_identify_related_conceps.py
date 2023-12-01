
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

cache = {}

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
        print ("Article " + prefix + str(i))
        response += "Article " + prefix + str(i) + "\n"
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
    query1 = input("What is the domain you want to learn:")
    field = input("Enter your personal field of research :")

    key = query1 + field
    if (key in cache):
        print (cache[key])
        continue
    if (field.strip() == ""):
        field = "same"
    response1 = collect_results(search(query1,an,model,30),sents_from_path1, path1)
    response2 = collect_results(search(query1,an2,model,10),sents_from_path2, path2)

    responseA = response2 + "\n" + response1
    if (len(responseA) > 70000):
        responseA = responseA[:70000]


   

    template = """ Take a deep breath. 
    You are an AI Assitant. 
    TASK DESCRIPTION: 
This task deals with facilitating effective research in a specific domain, particularly when the researcher may not be fully versed with the terminology used in that domain. Different fields and domains have their unique jargons, terminologies, and nomenclatures, which could serve as barriers for researchers coming from other fields. By identifying and providing a list of relevant search terms, the AI system can help overcome this challenge, aiding researchers in conducting thorough literature reviews, finding relevant papers, or understanding the current state of research in that domain.
For example, suppose a researcher with a background in computer science wants to explore research papers in the domain of microbiology, specifically regarding antibiotic resistance. They might not be familiar with specific terms used within the microbiology and antibiotic resistance research community. They might use terms like "drug resistance" which is a common term, but might not be aware of or use a more specific term like "beta-lactamase resistance" which refers to a particular type of antibiotic resistance.
An AI ASSISTANT could help by providing a list of relevant terms related to antibiotic resistance within the microbiology domain, making it easier for the researcher to search for and find relevant literature, datasets, and other resources. Such a solution could use natural language processing (NLP) techniques to extract and provide synonyms, related terms, or even hierarchies of terms from a corpus of domain-specific literature, thereby aiding the researcher in navigating the domain's terminology and conducting a more effective and thorough search.    Read and Analyze all the articles and identify the most important concepts in the FIELD of {query1}.
   
    FIELD: {query1}
    USER FIELD: {field}
    TOP ARTICLES: {responseA}
    INSTRUCTIONS:
  
    1. User wants to learn about the FIELD of {query1}.
    2. User is from the field of {field}.
    3. Read each Article and extract the most important concepts in the FIELD of {query1}.
    4. Generate a list of domain specific jargons and terminologies that can be used to search for articles in the FIELD of {query1}.
    5. For each jargon, generate a keyword, explain it with a definition and an example. 
    6. if the USER FIELD is NOT same, then for each jargon, include an example ANALOGY from the USER FIELD.
    7. Output should STRICTLY BE in JSON format following the SCHEMA. 
    8. Output should be a list of concepts in the order of importance to the FIELD of {query1}. Key concepts should be at the top of the list.
    9. List upto 100 CONCEPTS and atleast 10 CONCEPTS.
    Class Output:
        Keyword: String,
        Definition: String,
        Example: String,
        Analogy: String


    List the concepts in the order of importance to the FIELD of {query1}.
    For each concept, provide a list of keywords that can be used to search for articles in the FIELD of {query1}.
    """
    prompt = PromptTemplate(template=template, input_variables=["query1", "field", "responseA"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    try:
        output = llm_chain.run({'query1': query1, 'field': field, 'responseA':responseA})
        print (output)
        cache[key] = output
    except Exception as e:
        print (e)
        str_e = str(e)
        if "token" in str(e):
            print ("Token error")
            continue
