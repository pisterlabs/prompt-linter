from copy import deepcopy
from pprint import pp
import sys 
sys.path.insert(0, ".")

import pathlib
from utils.hash import get_hash
from nice_functions import *
from os.path import join
import os
from nice_functions import *
import tiktoken
import utils.config as config

from core.extensions import db
from core.openai_tools import load_api_keys
import openai

def print_summary_medium():
    load_api_keys()
    db.init_db()
    db.load_datasource("test_embedding_medium")
    docs = db.datasource.get(limit=10)
    pprint(docs)
    
def print_summary_basic():
    load_api_keys()
    db.init_db()
    db.load_datasource("test_embedding_basic")
    docs = db.datasource.get(limit=10)
    pprint(docs)


def simple_gpt(system_message, user_message):
    models_to_try = ["gpt-3.5-turbo-16k", "gpt-3.5-turbo"]
    for model_to_try in models_to_try:
        try:
            response = openai.ChatCompletion.create(
                model=model_to_try,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                temperature=1,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                # stream=True,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(red(model_to_try), "FAILED!")
            if model_to_try == models_to_try[-1]: raise(e)


def reduce_synopsis(synopsis, to_number_of_tokens):
    answer = simple_gpt(
        "You are a bot capable of summarize scientific articles", 
        rf"Please, can you summarize the following text returning no more than {to_number_of_tokens} tokens? The text to summarize is: {synopsis}")
    return answer


def get_db_summary():
    db_summary_path = join(pathlib.Path(__file__).parent.parent.resolve(), "db_summary", "test_embedding.txt")
    f = open(db_summary_path, "r")
    return f.read()

def create_db_summary():
    def summarize_block(docs):
        import re
        all_docs = "\n\n".join(docs)
        all_docs =  re.sub("Paper Link:.+?\n", "", all_docs, flags=re.DOTALL)
        summary = simple_gpt("""
        you are a system that process a list of papers.
        this list contains the titles, the authors, the publishing dates, and the links.
        you are capable of processing all these paper, and return:
        - the total number of papers
        - the main research area of the papers
        - the number of paper per research area.
        dont add filling text to the answer.
        use the paper title to guess the topic.
        the number of papers is given by the different titles of the papers, not by the links.
        """, all_docs)
        return(summary) 

    def summarize_all_blocks(blocks):
        all_blocks = [f"This is the beggining of a block:\n{el}\nThis is the end of a block\n\n" for el in blocks]
        all_blocks = "\n".join(all_blocks)
        summary = simple_gpt("""
        you are a system that process the summary of blocs of data. 
        each bloc's summary contains the total number of papers, the main topics of the papers, and the number of papers per topic. 
        all these papers mentioned in the blocs are part of the Quantum Networks Database (CQN database), and that is important. 
        you are capable of processing all these blocs, and give a summary of the Quantum Networks Database (CQN database).  
        the summary should include:
        - total papers
        - the most relevant and broad research area or topic of the papers  
        - more detailes research area or topic of the papers 
        - an approximate number of articles por per topic
        Start your suggested text with something like 'The Quantum Networks Database (CQN database) contains approximately'.
        Dont ever mention the word bloc.
        Use approximate numbers.
        """, all_blocks)
        return(summary) 

    load_api_keys()
    db.init_db()
    db.load_datasource("test_embedding_basic")
    print("loading db...")
    docs = db.datasource.get()
    summarize_block_of = 50
    docs = docs["documents"]
    blocks_summaries = []
    for i in range( int(len(docs)/summarize_block_of) + 1 ): 
        print(rf"Summarizing block {i}...")
        docs_block = docs[i*summarize_block_of:(i+1)*summarize_block_of]
        uid = get_hash(docs_block)
        block_summary = config.load(uid, default={}, ask=False )
        if block_summary == {}:
            block_summary = summarize_block(docs_block)
            config.save(uid, block_summary )
        pprint(block_summary)
        print("-"*100)
        blocks_summaries.append(block_summary)
    
    print("Making summary of all blocks...")
    db_summary = summarize_all_blocks(blocks_summaries)
    pprint("db_summary", db_summary)
    import pathlib
    from os.path import join
    db_summary_path = join(pathlib.Path(__file__).parent.parent.resolve(), "db_summary", "test_embedding.txt")
    f = open(db_summary_path, 'w')
    f.write(db_summary)
    
def create_embeddings_with_levels_of_information():
    load_api_keys()
    db.init_db()

    offset = 0
    limit = 1000
    while True:
        print(f"Getting documents {offset} to {offset+limit} ...")

        
        db.load_datasource("test_embedding")
        docs_full = db.datasource.get( offset=offset, limit=limit, include=["metadatas"])
        if docs_full["ids"] == []: break
        offset+=limit
        
        summary_keys = [{
            "name": "Paper Title",
            "regex": "title"
        },{
            "name": "Paper Authors",
            "regex": "author",
        },{
            "name": "Paper Published Data",
            "regex": "published"
        },{
            "name": "Paper Link",
            "regex": "links"
        },{
            "name": "Paper Summary",
            "regex": "summary"        
        }]
        
        keys_for_uid = ["Paper Title", "Paper Authors", "Paper Published Data"]
        
        docs_summarized = {}
        docs_metadatas = {}
        for doc_metadata in docs_full["metadatas"]:
            doc_summarized = get_doc_summary(doc_metadata, summary_keys)
            if doc_summarized=={}:continue
            str_for_id = [v for k, v in doc_summarized.items() if k in keys_for_uid]
            uid = get_hash(str_for_id)
            docs_summarized[uid] = doc_summarized
            docs_metadatas[uid] = doc_metadata
        
        pprint("Total documents:", len(docs_summarized))
        levels = ["medium", "basic"]
        
        for level in levels: 
            print()
            print(f"Generating {level} level resumen")
            db.load_datasource(rf"test_embedding_{level}")
            current_uids = db.datasource.get(include=[])["ids"]

            for uid, _doc_summarized in docs_summarized.items():
                doc_summarized = deepcopy(_doc_summarized)
                if uid in current_uids:
                    print(f"{green(uid)}: {doc_summarized['Paper Title'][0:50]}\n -> {blue('already added')}")
                    continue
                
                if level == "medium":
                    synopsis = doc_summarized.get("Paper Summary", "")
                    synopsis_tokens = len(tiktoken.get_encoding("cl100k_base").encode(synopsis))
                    if synopsis_tokens > 300:
                        print("Summary too loog... reducing...")
                        synopsis = reduce_synopsis(synopsis, to_number_of_tokens=300)
                        doc_summarized["Paper Summary"] = synopsis
                elif level == "basic":
                    doc_summarized.pop("Paper Summary")

                summarized_docs_string = stringify_doc_summary(doc_summarized)
                doc_metadata = docs_metadatas[uid]
                db.datasource.add(
                    ids=uid,
                    metadatas=doc_metadata,
                    documents=summarized_docs_string,
                )
                print(f"{green(uid)}: {doc_summarized['Paper Title'][0:50]}\n -> {green('added')}")


def get_keys_by_regex(regex, dict):
    keys = []
    import re
    for k,v in dict.items():
        if re.findall(regex, k, flags=re.IGNORECASE):
            keys.append(k)
    return keys
    
def get_doc_summary(doc_metadata, summary_keys):
    doc_summary = {}
    for summary_key in summary_keys:
        summary_key_name = summary_key["name"]
        summary_key_regex = summary_key["regex"]
        summary_key_function = summary_key.get("function", None)
        doc_metadata_keys = get_keys_by_regex(summary_key_regex, doc_metadata)
        if doc_metadata_keys:
            summary_key_value = get_values_from_keys(doc_metadata_keys, doc_metadata)
            if not summary_key_value: continue
            if summary_key_function == None:
                doc_summary[summary_key_name] = summary_key_value
            else:
                raise(Exception("To be implemented"))      
    return doc_summary
                        
def stringify_doc_summary(doc_summary):
    doc_summary_str = ""
    for k, v in doc_summary.items():
        if v: doc_summary_str+=f"{k}: {v}\n"
    return doc_summary_str

def get_values_from_keys(keys, dict, joiner = ", "):
    return joiner.join([ v for k,v in dict.items() if k in keys ])

