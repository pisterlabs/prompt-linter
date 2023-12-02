from model_dict import load_from_catalogue
from guidance_model import Summarization
import argparse
import pandas as pd
import csv
from tqdm import tqdm
import numpy as np
import os
import json
from run import make_initial_guide, find_penalty, penalty_guide, run
import guidance
import pdb


def main(args):
    
    output_path = f'/Workspace/ky/SharedTask2023/results/summarization/{args.model_name}/{args.data_type}/{args.exp_name}/{args.trial_no}'
    os.makedirs(output_path, exist_ok=True)

    
    with open(os.path.join(args.prompt_path,"score_prompt.txt"), 'r', encoding='utf-8') as file:
        score_prompt = file.read()
    with open(os.path.join(args.prompt_path,"guide_prompt.txt"), 'r', encoding='utf-8') as file:
        guide_prompt = file.read()
    with open(os.path.join(args.prompt_path,"penalty_prompt.txt"), 'r', encoding='utf-8') as file:
        penalty_prompt = file.read()
    with open(os.path.join(args.prompt_path,"score_options.json"), 'r', encoding='utf-8') as file:
        score_options = json.load(file)

    
    model_path = os.path.join('/Workspace/jh/SharedTask2023/models',args.model_name)
    model, tokenizer, u_prompt, a_prompt = load_from_catalogue(args.model_name, model_path, args.device)
    print('model loaded')
    
    BPG = Summarization(model=model, 
                tokenizer=tokenizer, 
                score_prompt=score_prompt,
                guide_prompt=guide_prompt,
                penalty_prompt=penalty_prompt,
                score_options=score_options,
                prompt_placeholder=u_prompt,
                response_placeholder=a_prompt
                )

    data = pd.read_csv(args.data_path, delimiter="\t", quoting=csv.QUOTE_NONE)
    
    consistency, coherence, relevance = make_initial_guide(BPG)
        
    penalty_tuple = (1, 1, 1)
    
    idx = 0 
    
    results = None
    
    while np.sum(penalty_tuple):
        # Ugly fix for memory leak; perhaps with the guidance module
        if BPG:
            del BPG
        BPG = Summarization(model=model, 
                    tokenizer=tokenizer, 
                    score_prompt=score_prompt,
                    guide_prompt=guide_prompt,
                    penalty_prompt=penalty_prompt,
                    score_options=score_options,
                    prompt_placeholder=u_prompt,
                    response_placeholder=a_prompt,
                    )
        
        print(f'{idx}th iteration')
        
        results = run(consistency=consistency,
                        coherence=coherence, 
                        relevance=relevance, 
                        BPG=BPG, 
                        data=data,
                        score_options=score_options,
                        score_prompt=score_prompt,
                        guide_prompt=guide_prompt,
                        penalty_prompt=penalty_prompt,
                        penalty_tuple=penalty_tuple,
                        idx=idx,
                        previous_results=results,
                        )
        
        # pdb.set_trace()
        with open(f'{output_path}/{idx}.json','w') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        
        penalty_tuple = find_penalty(results)
        # pdb.set_trace()
        consistency, coherence, relevance = penalty_guide(BPG=BPG,
                                                            con=consistency,
                                                            coh=coherence,
                                                            rel=relevance,
                                                            penalty_tuple=penalty_tuple,
                                                            penalty_prompt=penalty_prompt,
                                                            )
        # pdb.set_trace()
        idx += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--trial_no', type=str)
    parser.add_argument('--model_name', type=str, default='OpenOrca-Platypus2-13B')
    parser.add_argument('--data_path', type=str, default='/Workspace/jh/SharedTask2023/data/summarization/train_summarization.tsv')
    parser.add_argument('--prompt_path', type=str, default='/Workspace/ky/SharedTask2023/prompt/ver4')
    parser.add_argument('--data_type', type=str, default='train')
    parser.add_argument('--device', type=str, default='cuda:1')
    
    args = parser.parse_args()
    
    main(args)
    
    