import argparse

import openai
import pandas as pd
import requests
from collections import defaultdict

import re
import os
import string
import json

from model_q import model_q
from model_a import model_a
from model_r import model_r
import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #parser.add_argument("--openai-api-key",type=str,required=True,help="OpenAI API Key.")
    #parser.add_argument("--bing-api-key",type=str,required=True,help="Bing API Key.")
    parser.add_argument("--top-k-search-results",type=int,default=5,help="Number of search results to use for each query.")
    parser.add_argument("--dataset",type=str,default='triviaQA',help="Dataset type: hotpotQA, naturalQA, triviaQA")
    parser.add_argument("--n",type=int,default='5',help="num questions from datasets to run on")
    parser.add_argument("--model_q_mode",type=str,default='no_rewrite',help="Mode for Model Q")
    parser.add_argument("--model_r_mode",type=str,default='RM',help="Mode for Model R")
    parser.add_argument("--model_a_mode",type=str,default='answer',help="Mode for Model A")

    parser.add_argument("--wiki_only",type=str,default='False',help="should you restrict bing search to wiki")
    parser.add_argument("--entire_doc",type=str,default='False',help="should you grab entire html doc from wiki")


    args = parser.parse_args()
    match_num=0

    # DATASET PROCESSING 
    dataset = utils.process_dataset(args.dataset, n=args.n)

    results = defaultdict(dict)
    results['dataset_name'] = args.dataset
    results['data'] = {}
    results['model_q_mode']=args.model_q_mode
    results['model_a_mode']=args.model_a_mode
    results['model_r_mode']=args.model_r_mode
    results['wiki_only']=args.wiki_only
    results['entire_doc']=args.entire_doc
    results['k']=args.top_k_search_results

    args.wiki_only = args.wiki_only=='True'
    args.entire_doc = args.entire_doc=='True'

    for data in dataset:
        question, answer_gt = data['question'], data['answer']

        # MODEL Q
        query = model_q(question, mode=args.model_q_mode)

        # MODEL R 
        docs = model_r(query, top_k_search_results=args.top_k_search_results, mode=args.model_r_mode, wiki_only=args.wiki_only, entire_doc=args.entire_doc)

        # MODEL A
        predicted_answer = model_a(query, docs, mode=args.model_a_mode)

        # store results
        results['data'][question] = {'answer_gt': answer_gt, 'answer_pred': predicted_answer, 'query': query, 'docs': docs}    

        # save results

        if not os.path.isdir(os.getcwd() + '/data/{}'.format(args.dataset)): os.mkdir(os.getcwd() + '/data/{}'.format(args.dataset))
        if not os.path.isdir(os.getcwd() + '/data/{}/n_{}/'.format(args.dataset, args.n)): os.mkdir(os.getcwd() + '/data/{}/n_{}'.format(args.dataset, args.n))
        if not os.path.isdir(os.getcwd() + '/data/{}/n_{}/top{}docs/'.format(args.dataset, args.n, args.top_k_search_results)): os.mkdir(os.getcwd() + '/data/{}/n_{}/top{}docs/'.format(args.dataset, args.n, args.top_k_search_results))
        
        with open(os.getcwd() + '/data/{}/n_{}/top{}docs/{}-{}-{}-wiki_only_{}-entire_doc_{}.json'.format(args.dataset, args.n, args.top_k_search_results, args.model_q_mode, args.model_r_mode, args.model_a_mode, args.wiki_only, args.entire_doc), 'w') as fp:
            json.dump(results, fp)

    # EVAL STUFF
    #match_num += substring_match_score(prediction, answer_gt)











