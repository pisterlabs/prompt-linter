'''
This baseline implements the direct assessment prompt by
Kocmi and Federmann, Large Language Models Are State-of-the-Art Evaluators of Translation Quality. ArXiv: 2302.14520
for open source LLMs with MT and summarization
'''

import guidance
import torch
from model_dict import load_from_catalogue

import argparse
import pandas as pd
import csv
from tqdm import tqdm
import numpy as np
import os
import pdb
import json

class Summarization:
    def __init__(self, 
                 model, 
                 tokenizer,
                 guide_prompt, 
                 score_prompt,
                 penalty_prompt,
                 score_options,
                 prompt_placeholder,
                 response_placeholder,
                 **kwargs):
        self.g_model = guidance.llms.Transformers(
            model, tokenizer=tokenizer, trust_remote_code=True, **kwargs
        )
        self.score_prompt =  score_prompt
        self.guide_prompt = guide_prompt
        self.penalty_prompt = penalty_prompt
        self.score_options = score_options
        self.prompt_placeholder = prompt_placeholder
        self.response_placeholder = response_placeholder
        guidance.llms.Transformers.cache.clear()
        guidance.llm = self.g_model


    def set_model(self, model):
        self.model = model
        guidance.llms.Transformers.cache.clear()
        
    # def cache_clear(self):
    #     self.guidance.llm.cache.clear()
    #     torch.cuda.empty_cache()

    def do_guide(
        self,
        penalty_prompt=None,
        previous_answer=None,
        aspect=None,
        score=None,
        verbose=False,
    ):
        guide_program = guidance('''{{input_prompt}}''')
        response = guide_program(
            input_prompt=self.guide_prompt,
            penalty_prompt=None,
            previous_answer=None,
            aspect=aspect,
            score_options=self.score_options,
            prompt_placeholder=self.prompt_placeholder,
            response_placeholder=self.response_placeholder,
            silent=True,
            caching=False,
            )()
        
        # pdb.set_trace()
        guide = response['answer']
        
        if verbose:
            print(self.guide)
        # guidance.llm.cache.clear()
        torch.cuda.empty_cache()
        guidance.llms.Transformers.cache.clear()
        
        return guide
    
    def do_penalty(
        self,
        penalty_prompt=None,
        previous_answer=None,
        aspect=None,
        score=None,
        verbose=False
    ):
    
        guide_program = guidance('''{{input_prompt}}}''')
        response = guide_program(
            input_prompt=self.guide_prompt,
            penalty_prompt=penalty_prompt,
            previous_answer=previous_answer,
            aspect=aspect,
            score_options=self.score_options,
            prompt_placeholder=self.prompt_placeholder,
            response_placeholder=self.response_placeholder,
            silent=True,
            caching=False,
            )()
        
        guide = response['answer']
        
        if verbose:
            print(self.guide)
        # guidance.llm.cache.clear()
        torch.cuda.empty_cache()
        guidance.llms.Transformers.cache.clear()
        
        return guide
    
    def do_score(
        self,
        score_prompt=None,
        guide=None,
        source=None,
        summary=None,
        score_options=None,
    ):
    
        results = {}
        score_program = guidance('''{{input_prompt}}''')
        response = score_program(
            input_prompt=self.score_prompt,
            guide=guide,
            source=source,
            summary=summary,
            score_options=self.score_options,
            options=list(self.score_options.values()),
            prompt_placeholder=self.prompt_placeholder,
            response_placeholder=self.response_placeholder,
            silent=True,
            caching=False
            )()

    
        score = np.sum( np.exp(list(response['logprobs'].values())) * np.array([5,3,1]) )
        # if verbose:
        #     print(self.prompt)
        # guidance.llm.cache.clear()
        torch.cuda.empty_cache()
        guidance.llms.Transformers.cache.clear()
        
        return score
    