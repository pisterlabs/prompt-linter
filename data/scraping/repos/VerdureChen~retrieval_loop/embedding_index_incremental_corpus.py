from langchain.document_loaders import HuggingFaceDatasetLoader
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os
from elastic_bm25_search_with_metadata import ElasticSearchBM25Retriever
import elasticsearch
from langchain.vectorstores import ElasticsearchStore
from langchain.vectorstores.utils import DistanceStrategy
from retrieve_methods import Retrieval
import argparse
import json
from tqdm import tqdm
import torch
import sys
from math import ceil
sys.stdout.flush()

def get_args():
    # get config_file_path, which is the path to the config file
    # config file formatted as a json file:
    # {
    #   "new_text_file": "../../data_v2/zero_gen_data/DPR/nq-test-gen-baichuan2-13b-chat.jsonl",
    #   "retrieval_method": "DPR", # BM25, DPR, Contriever, RetroMAE, all-mpnet, BGE, LLM-Embedder
    #   "index_name": "DPR_faiss_index",
    #   "index_path": "../../data_v2/indexes",
    #   "page_content_column": "question"
    # }
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file_path", type=str, required=True)
    args = parser.parse_args()
    # read config file
    config_file_path = args.config_file_path
    with open(config_file_path, "r", encoding='utf-8') as f:
        config = json.load(f)
    print(f'config: {config}')

    return config


def load_retrieval_embeddings(retrieval_model, normalize_embeddings=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': normalize_embeddings,
                     'batch_size': 512,
                     'show_progress_bar': True}
    embeddings = HuggingFaceEmbeddings(model_name=retrieval_model, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
    return embeddings


def main(new_text_file, page_content_column, retrieval_model, index_name, index_path, normalize_embeddings, index_exists):

    # load the new text file
    loader = HuggingFaceDatasetLoader('json', data_files=new_text_file,
                                      page_content_column=page_content_column)
    new_text = loader.load()
    print(f'lenght of new text: {len(new_text)}')
    print(f'new text format: {new_text[0]}')

    # map retrieval model names: DPR, Contriever, RetroMAE, all-mpnet, BGE, LLM-Embedder
    instruction = ''
    if 'DPR' in retrieval_model.upper():
        retrieval_model = '../../ret_model/DPR/facebook-dpr-ctx_encoder-multiset-base'
    elif 'CONTRIEVER' in retrieval_model.upper():
        retrieval_model = '../../ret_model/contriever-base-msmarco'
    elif 'RETROMAE' in retrieval_model.upper():
        retrieval_model = '../../ret_model/RetroMAE_BEIR'
    elif 'ALL-MPNET' in retrieval_model.upper():
        retrieval_model = '../../ret_model/all-mpnet-base-v2'
    elif 'BGE-LARGE' in retrieval_model.upper():
        retrieval_model = '../../ret_model/bge-large-en-v1.5'
    elif 'BGE-BASE' in retrieval_model.upper():
        retrieval_model = '../../ret_model/bge-base-en-v1.5'
    elif 'LLM-EMBEDDER' in retrieval_model.upper():
        retrieval_model = '../../ret_model/llm-embedder'
        instruction = "Represent this document for retrieval: "
    elif 'BM25' in retrieval_model.upper():
        retrieval_model = 'BM25'
    else:
        raise ValueError(f'unknown retrieval model: {retrieval_model}')

    # load the retrieval embeddings
    if retrieval_model != "BM25":
        embeddings = load_retrieval_embeddings(retrieval_model, normalize_embeddings=normalize_embeddings)
        print(f'loaded retrieval embeddings: {retrieval_model}')
    else:
        print('Please make sure you have started elastic search server')
        elasticsearch_url = "http://0.0.0.0:9978"

    # process the new text adding instruction
    if instruction != '':
        for i in tqdm(range(len(new_text)), desc='adding instruction to new text'):
            new_text[i].page_content = instruction + new_text[i].page_content
    print(f'processed new text format: {new_text[0]}')

    # check if the index exists
    if index_exists:
        print(f'index: {index_name} exists')
        # load the index
        if retrieval_model != "BM25":
            index_p = os.path.join(index_path, index_name)
            index = FAISS.load_local(index_p, embeddings)
            print(f'loaded {retrieval_model} index: {index_name}, length: {len(index.docstore._dict)}')
            print(f'adding new text to index')
            db2 = FAISS.from_documents(new_text, embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
            index.merge_from(db2)
            index.save_local(index_p)
            print(f'added {len(db2.docstore._dict)} new text to index: {index_name}, length: {len(index.docstore._dict)}')
        else:
            index = ElasticSearchBM25Retriever.create(elasticsearch_url, index_name)
            index_size = index.get_document_count()
            print(f'loaded {retrieval_model} index: {index_name}, length: {index_size}')
            print(f'adding new text to index')
            index.add_texts(new_text)
            index_size = index.get_document_count()
            print(f'added {len(new_text)} new text to index: {index_name}, length: {index_size}')

    else:
        print(f'index: {index_name} does not exist')
        print(f'creating {retrieval_model} index')
        # create the index
        if retrieval_model != "BM25":
            index_p = os.path.join(index_path, index_name)
            index = FAISS.from_documents(new_text, embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
            index.save_local(index_p)
            index_size = len(index.docstore._dict)
        else:
            index = ElasticSearchBM25Retriever.create(elasticsearch_url, index_name, overwrite_existing_index=True)
            index_size = index.get_document_count()
            print(f'loaded {retrieval_model} index: {index_name}, length: {index_size}')
            print(f'adding new text to index')
            index.add_texts(new_text)
            index_size = index.get_document_count()
            print(f'added {len(new_text)} new text to index: {index_name}, length: {index_size}')


        print(f'created {retrieval_model} index: {index_name}, length: {index_size}')


import linecache

def count_lines(filename):
    count = len(linecache.getlines(filename))
    return count


if __name__ == '__main__':
    config = get_args()
    new_text_file = config["new_text_file"]
    page_content_column = config["page_content_column"]
    retrieval_model = config["retrieval_model"]
    normalize_embeddings = config["normalize_embeddings"]
    index_name = config["index_name"]
    index_path = config["index_path"]
    index_exists = config["index_exists"]
    # json:
    # {
    #   "new_text_file": "../../data_v2/zero_gen_data/DPR/nq-test-gen-baichuan2-13b-chat.jsonl",
    #   "index_name": "DPR_faiss_index",
    #   "index_path": "../../data_v2/indexes",
    #   "page_content_column": "contents"
    #   "index_exists": True,
    #   "normalize_embeddings": False,
    #   "query_file": "../../data_v2/zero_gen_data/DPR/nq-test-gen-baichuan2-13b-chat.jsonl",
    #   "query_page_content_column": "question",
    #   "retrieval_model": "DPR",  # BM25, DPR, Contriever, RetroMAE, all-mpnet, BGE, LLM-Embedder
    #   "output_file": "../../data_v2/zero_gen_data/DPR/nq-test-gen-baichuan2-13b-chat.jsonl"
    # }

    # count lines in the new text file, if it is larger than 2000000, then split it into 2000000 lines each
    # count the lines
    print(f'counting lines in the new text file: {new_text_file}')
    line_count = count_lines(new_text_file)
    print(f'line count: {line_count}')
    # count the number of files
    file_count = ceil(line_count/2000000)
    print(f'file count: {file_count}')
    # split the file and save them into a list
    new_text_file_list = []
    for i in range(file_count):
        new_text_file_list.append(f'{new_text_file}_{retrieval_model}_{i}.jsonl')
    # split the file
    with open(new_text_file, 'r', encoding='utf-8') as f:
        for i in tqdm(range(file_count), desc='splitting the new text file'):
            with open(new_text_file_list[i], 'w', encoding='utf-8') as f2:
                for j in range(2000000):
                    line = f.readline()
                    f2.write(line)
    # load the new text file
    for i in tqdm(range(file_count), desc='processing the new text file'):
        if i!=0:
            index_exists = True
        new_text_file = new_text_file_list[i]
        print(f'processing file: {new_text_file}')
        main(new_text_file, page_content_column, retrieval_model, index_name, index_path, normalize_embeddings, index_exists)

    #remove the split files
    for i in range(file_count):
        os.remove(new_text_file_list[i])

    query_files = config["query_files"]
    page_content_column = config["query_page_content_column"]
    retrieval_model = config["retrieval_model"]
    index_name = config["index_name"]
    index_path = config["index_path"]
    normalize_embeddings = config["normalize_embeddings"]
    output_files = config["output_files"]

    # test the index
    # load the test query file

    Retrieval(query_files, page_content_column, retrieval_model, index_name, index_path, normalize_embeddings,
              output_files)

