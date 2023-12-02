from django.shortcuts import render

# Create your views here.
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import json
from fastapi.encoders import jsonable_encoder
import re

import os
import sys
import openai
import backoff
from functools import partial
import re
import numpy as np
# import file
from autobot.utils.prompts import *
import autobot.utils.embeddings

EMBEDDING_PICKLE_PATH='./autobot/model/sentence_embedding_func.pkl'
VECTOR_SAVE_PATH = './autobot/chroma_database'
COHERE_API_KEY = os.environ['cohere_key']
OPENAI_KEY = os.environ['OPENAI_API_KEY']
openai.api_key = OPENAI_KEY

# user sends files to upload
@api_view(['POST'])
def process_file(request):
    if request.method == 'POST':
        # gets files
        files = request.FILES.getlist('files')
        if not files or len(files) < 1:
            return Response('No files received', status=status.HTTP_400_BAD_REQUEST)
    
        # turns files into documents
        print('getting documents')
        documents = autobot.utils.embeddings.doc_load(files)

        # create and save embeddings
        print('creating embedding')
        autobot.utils.embeddings.save_chroma_using_embedding(documents,
                            EMBEDDING_PICKLE_PATH,
                            VECTOR_SAVE_PATH)
        print('ready to query')
        return Response({'result': 'success'})



@api_view(['POST'])
def chat(request):
    if request.method == 'POST':
        
        # gets query
        query = request.data.get('input_string', '')

        # gets top ranked documents
        print('loading documents related to query')
        top_docs = autobot.utils.embeddings.load_chroma_with_query_with_compressor(VECTOR_SAVE_PATH,
                       EMBEDDING_PICKLE_PATH,
                       query, COHERE_API_KEY)

        # converts documents to necessary string
        tot_sources = {}
        for doc in top_docs:
            tot_sources[doc.metadata['source']] = doc.page_content

        tot_string = json.dumps(tot_sources)
        print(tot_string)
        # return Response({'answer': 'random'})

        # runs tree of thought
        x = {'question': query, 'sources': tot_string}
        print('running tree of thought')
        try:
            responsegen=run(x)
        except:
            print("error")

        return Response({'answer': responsegen})


def remove_non_alphanumeric(text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return cleaned_text

steps = 1

completion_tokens = prompt_tokens = 0

@backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(**kwargs):
    print("in completions_with_backoff")
    global res
    try:
        res=openai.ChatCompletion.create(**kwargs)
    except Exception as error:
        print(error)
    return res

def gpt(prompt, model="gpt-3.5-turbo", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    
def chatgpt(messages, model="gpt-3.5-turbo", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    print("in chatgpt")
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        
        res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
        
        print("out  completions_with_backoff")
        outputs.extend([choice["message"]["content"] for choice in res["choices"]])
        # log completion tokens
        completion_tokens += res["usage"]["completion_tokens"]
        prompt_tokens += res["usage"]["prompt_tokens"]
    return outputs
    
def gpt_usage(backend="gpt-3.5-turbo"):
    global completion_tokens, prompt_tokens
    if backend == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo":
        cost = (completion_tokens + prompt_tokens) / 1000 * 0.0002
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}


class PromptTree:
        
    def __init__(self):
        self.y = ''
        self.steps = 2
        self.stops = []
    
    def generate_thoughts(self, state, question, sources, stop=None):
        # print("STATE", state)
        # print("QUESTION", question)
        # print("SOURCES", sources)
        prompt = cot_prompt.format(question=question, sources=sources) + state
        print("in generate_thoughts")
        thoughts = gpt(prompt, n=3, stop=stop)
        return [state + _ for _ in thoughts]

    def generate_vote(self, thoughts):
            prompt = vote_prompt
            for i, y in enumerate(thoughts, 1):
                # y = y.replace('Plan:\n', '')
                # TODO: truncate the plan part?
                prompt += f'Choice {i}:\n{y}\n'

            vote_outputs = gpt(prompt, n=3, stop=None)

            vote_results = [0] * len(thoughts)
            for vote_output in vote_outputs:
                pattern = r".*best choice is .*(\d+).*"
                match = re.match(pattern, vote_output, re.DOTALL)
                if match:
                    vote = int(match.groups()[0]) - 1
                    if vote in range(len(thoughts)):
                        vote_results[vote] += 1
                else:
                    print(f'vote no match: {[vote_output]}')
            
            return np.argmax(vote_results)




    def solve(self, x):
        '''
        Given an input, uses tree of thought prompting to generate output to answer.
        '''
        
        new_ys = self.generate_thoughts(self.y, *x.values(), '\nSources:')
        best_idx = self.generate_vote(new_ys)
        self.y = new_ys[best_idx]
        new_ys = self.generate_thoughts(self.y, *x.values(), '\nAnswer:')
        best_idx = self.generate_vote(new_ys)
        self.y = new_ys[best_idx]
        new_ys = self.generate_thoughts(self.y, *x.values())
        best_idx = self.generate_vote(new_ys)
        self.y = new_ys[best_idx]

        # Extract the "Answer" part using slicing
        answer_start = self.y.find("Answer:")
        answer_end = self.y.find("Reasoning:")
        answer = self.y[answer_start + len("Answer:"):answer_end].strip()

        return answer

def run(x):
    global gpt
    gpt = partial(gpt, model='gpt-3.5-turbo', temperature=1.0)
    tree_of_thought = PromptTree()
    return tree_of_thought.solve(x)