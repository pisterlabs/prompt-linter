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
                 score_prompt, 
                 summarization_prompt, 
                 summarization_temperature,
                 explain_temperature,
                 answer_temperature,
                 **kwargs):
        self.model = guidance.llms.Transformers(
            model, tokenizer=tokenizer, trust_remote_code=True, **kwargs
        )
        self.score_prompt =  score_prompt
        self.summarization_prompt = summarization_prompt
        self.summarization_temperature = summarization_temperature
        self.explain_temperature = explain_temperature
        self.answer_temperature = answer_temperature
        # self.score_prompt = os.path.join(prompt_path, "score_prompt.txt")
        # self.summarization_prompt = os.path.join(prompt_path, "summarization_prompt.txt")

    def set_model(self, model):
        self.model = model
        guidance.llms.Transformers.cache.clear()
        
    # def cache_clear(self):
    #     self.guidance.llm.cache.clear()
    #     torch.cuda.empty_cache()

    def summarization_prompt_model(
        self,
        gt,
        hyp,
        prompt_placeholder=None,
        response_placeholder=None,
        verbose=False
    ):
    
        summarzation_program = guidance(self.summarization_prompt, llm=self.model)
        response = summarzation_program(
            gt=gt,
            hyp=hyp,
            summarization_temperature=self.summarization_temperature,
            prompt_placeholder=prompt_placeholder,
            response_placeholder=response_placeholder,
            silent=True
            )()
        
        summarization = response['summarization']
        
        summarization = filtering(summarization)

        if verbose:
            print(self.prompt)
        # guidance.llm.cache.clear()
        torch.cuda.empty_cache()
        guidance.llms.Transformers.cache.clear()
        return summarization
    
    def score_prompt_model(
        self,
        summary,
        aspect,
        prompt_placeholder=None,
        response_placeholder=None,
        verbose=False
    ):
    
        results = {}
        score_program = guidance(self.score_prompt, llm=self.model)
        response = score_program(
            summary=summary,
            aspect=aspect,
            explain_temperature=self.explain_temperature,
            answer_temperature=self.answer_temperature,
            prompt_placeholder=prompt_placeholder,
            response_placeholder=response_placeholder,
            silent=True
            )()

        # if verbose:
        #     print(self.prompt)
        # guidance.llm.cache.clear()
        torch.cuda.empty_cache()
        guidance.llms.Transformers.cache.clear()
        return response
    


def main(args):
    
    output_path = f'/Workspace/ky/SharedTask2023/results/summarization/{args.model_name}/{args.data_type}/{str(args.summarization_temperature)}_{str(args.explain_temperature)}_{str(args.answer_temperature)}_{args.exp_name}'
    os.makedirs(output_path, exist_ok=True)

    
    with open(os.path.join(args.prompt_path,"score_prompt.txt"), 'r', encoding='utf-8') as file:
        score_prompt = file.read()
        
    with open(os.path.join(args.prompt_path,"summarization_prompt.txt"), 'r', encoding='utf-8') as file:
        summarization_prompt = file.read()
        
    model_path = os.path.join('/Workspace/jh/SharedTask2023/models',args.model_name)
    model, tokenizer, u_prompt, a_prompt = load_from_catalogue(args.model_name, model_path, args.device)
        
    BPG = Summarization(model=model, 
                        tokenizer=tokenizer, 
                        score_prompt=score_prompt,
                        summarization_prompt=summarization_prompt,
                        summarization_temperature=args.summarization_temperature,
                        explain_temperature=args.explain_temperature,
                        answer_temperature=args.answer_temperature,
                        )

    
    data = pd.read_csv(args.data_path, delimiter="\t", quoting=csv.QUOTE_NONE)
    
    real_final_results = []
    
    for i in tqdm(range(len(data))):
        final_results = []
        try:
            aspects = BPG.summarization_prompt_model(
                gt=data["SRC"][i],
                hyp=data["HYP"][i],
                
                prompt_placeholder=u_prompt,
                response_placeholder=a_prompt
            )
            
            results = {'summary':[],
                        'aspect':[],
                        'explain':[],
                        'answer':[],
                        'score':[],
            }
            
            
            for aspect in aspects:
                response = BPG.score_prompt_model(
                    summary=data["HYP"][i],
                    aspect=aspect,
                    prompt_placeholder=u_prompt,
                    response_placeholder=a_prompt,
                    verbose=False
                )
                
                results['summary'].append(response['summary'])
                results['aspect'].append(response['aspect'])
                results['explain'].append(response['explain'])
                results['answer'].append(response['answer'])
                results['score'].append(int(response['score']))
                del response
                
            del aspects
            
            results['mean_score'] = np.mean(results['score'])
            results['gt_score'] = data["Score"][i]
            
            with open(f'{output_path}/{i}.json','w') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            
            final_results.append(results)
            real_final_results.append(results)
            
        except KeyboardInterrupt:
            exit()
        
        except:
            pass
    
    real_final_results = pd.DataFrame(real_final_results)
    
    real_final_results.to_csv(output_path+'train_result.csv', index=False)
    
    # if args.data_type == 'train':
    #     final_results['mean_score'].to_csv(args.output_path + ".seg.scores", header=False, index=False)

def filtering(input):
    return list(filter(None, input.split('\n')))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--model_name', type=str, default='Nous-Hermes-13b')
    # parser.add_argument('--model_path', type=str, default='/Workspace/jh/SharedTask2023/models/Nous-Hermes-13b')
    parser.add_argument('--data_path', type=str, default='/Workspace/jh/SharedTask2023/data/summarization/train_summarization.tsv')
    parser.add_argument('--summarization_temperature', type=float, default=0.7)
    parser.add_argument('--explain_temperature', type=float, default=1)
    parser.add_argument('--answer_temperature', type=float, default=1)
    parser.add_argument('--prompt_path', type=str, default='/Workspace/ky/SharedTask2023/prompt/dynamic_summary')
    parser.add_argument('--data_type', type=str, default='train')
    parser.add_argument('--device', type=str, default='cuda:1')
    
    args = parser.parse_args()
    
    main(args)
    
    