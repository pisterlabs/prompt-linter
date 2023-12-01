import json
import argparse
import evaluate
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from typing import Iterable
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from finetune import get_model_and_tokenizer
from evaluate_summarization import evaluate_hf_model
import transformers
import torch
    
#-----------------------
# Main Function
#-----------------------
def main():
    
    #-------------------
    # parameters
    #-------------------    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='True')
    parser.add_argument('--dataset', type=str, default='beanham/wikiqa')
    parser.add_argument('--input_column', type=str, default='Question')
    parser.add_argument('--target_column', type=str, default='Sentence')
    parser.add_argument('--start_prompt', type=str, default='### Answer the following question: ')
    parser.add_argument('--end_prompt', type=str, default='### Begin answering: ')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to mount the model on.')
    args = parser.parse_args()
  
    #-------------------
    # load data
    #-------------------
    print('Getting data...')
    train_data = load_dataset(args.dataset, split='train')
    validation_data = load_dataset(args.dataset, split='validation')
    test_data = load_dataset(args.dataset, split='test')
    
    #-------------------
    # load summarizer
    #-------------------
    print('Getting model and tokenizer...')
    model, tokenizer = get_model_and_tokenizer(args.model_id,
                                               gradient_checkpointing=False,
                                               quantization_type='4bit',
                                               device=args.device)
    
    #--------------
    # inference
    #--------------
    model.eval()
    #model.to(args.device)
    
    print('Evaluating model on ROUGE, BLEU, and BERTScore...')
    model_outputs, metrics = evaluate_hf_model(model, 
                                tokenizer, 
                                test_data, 
                                input_column=args.input_column,
                                target_column=args.target_column,
                                max_samples=len(test_data),
                                start_prompt=args.start_prompt,
                                end_prompt=args.end_prompt)
    for k, v in metrics.items():
        print(f'{k}: {v}')
        
    np.save(f"{args.model_id.split('/')[1]}_pretrained_model_outputs.npy", model_outputs)
    
if __name__ == "__main__":
    main()
