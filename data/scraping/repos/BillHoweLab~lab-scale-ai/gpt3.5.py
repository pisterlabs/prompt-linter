import json
import argparse
import evaluate
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from typing import Iterable
from datasets import load_dataset

def compute_summarization_metrics(predictions: Iterable, 
                                  references: Iterable,
                                  rouge: bool=True,
                                  bleu: bool=True,
                                  bertscore: bool=True) -> dict:
    """
    Compute ROUGE, BLEU, and BERTscore metrics for a set of predictions and references.
    """

    metric_results = {}

    if rouge:
        rouge = evaluate.load('rouge')

        # Compute ROUGE metrics at the summary level, using the 'rouge1', 'rouge2', and 'rougeL' metrics, aggregating the results
        rouge_results = rouge.compute(predictions=predictions, 
                                    references=references, 
                                    use_aggregator=True)

        # Store the results in the metric_results dictionary
        metric_results['rouge'] = rouge_results
    
    else:
        metric_results['rouge'] = None

    if bleu:
        bleu = evaluate.load('bleu')

        # Compute BLEU metrics at the summary level
        bleu_results = bleu.compute(predictions=predictions, 
                                    references=references)
        
        # Store the results in the metric_results dictionary
        metric_results['bleu'] = bleu_results
    
    else:
        metric_results['bleu'] = None

    if bertscore:
        bertscore = evaluate.load('bertscore')

        # Compute BERTscore metric, using distilbert-base-uncased as the reference model, and averaging the results
        bertscore_results = bertscore.compute(predictions=predictions, 
                                                    references=references, 
                                                    lang='en', 
                                                    model_type="distilbert-base-uncased")
        
        # Store the results in the metric_results dictionary
        metric_results['bertscore'] = {k: np.mean(v) for k, v in bertscore_results.items() if k in ['precision', 'recall', 'f1']}
    
    else:
        metric_results['bertscore'] = None

    return metric_results

def zeroshot(test_data, client):
    
    #--------------
    # inference
    #--------------
    predictions = []
    gts = []
    for i in tqdm(range(len(test_data))):
        
        system = 'Answer the following question: \n'
        dialogue = test_data[i]['Question'] 
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": dialogue},
            ],
            temperature=0,
            max_tokens=512,
            top_p=1
        )
        predictions.append(response.choices[0].message.content)
        gts.append(test_data[i]['Sentence'])

    np.save('gpt_zeroshot_outputs.npy', predictions)
    results = compute_summarization_metrics(predictions, gts)    
    with open('zeroshot-results.json', 'w') as f:
        json.dump(results, f)
    f.close()

def oneshot(train_data, test_data, client):
    #--------------
    # inference
    #--------------
    predictions = []
    gts = []
    for i in tqdm(range(len(test_data))):
        system = 'Summarize the following conversation\n'
        dialogue_example = train_data[0]['dialogue']
        summary_example = train_data[0]['section_text']
        dialogue = test_data[i]['dialogue']
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": dialogue_example},
                {"role": "assistant", "content": summary_example},
                {"role": "user", "content": dialogue}
            ],
            temperature=0,
            max_tokens=512,
            top_p=1
        )
        predictions.append(response.choices[0].message.content)
        gts.append(test_data[i]['section_text'])
        
    results = compute_summarization_metrics(predictions, gts)    
    with open('oneshot-results.json', 'w') as f:
        json.dump(results, f)
    f.close()
        
def fewshot(train_data, test_data, client):
    #--------------
    # inference
    #--------------
    predictions = []
    gts = []
    for i in tqdm(range(len(test_data))):
        system = 'Summarize the following conversation\n'
        dialogue_example_1 = train_data[0]['dialogue']
        summary_example_1 = train_data[0]['section_text']
        dialogue_example_2 = train_data[1]['dialogue']
        summary_example_2 = train_data[1]['section_text']
        dialogue = test_data[i]['dialogue']
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": dialogue_example_1},
                {"role": "assistant", "content": summary_example_1},
                {"role": "user", "content": dialogue_example_2},
                {"role": "assistant", "content": summary_example_2},
                {"role": "user", "content": dialogue}
            ],
            temperature=0,
            max_tokens=512,
            top_p=1
        )
        predictions.append(response.choices[0].message.content)
        gts.append(test_data[i]['section_text'])
        
    results = compute_summarization_metrics(predictions, gts)
    with open('fewshot-results.json', 'w') as f:
        json.dump(results, f)
    f.close()
#-----------------------
# Main Function
#-----------------------
def main():
    
    #-------------------
    # parameters
    #-------------------    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='beanham/wikiqa')  
    parser.add_argument('--shottype', type=str, default='True')
    parser.add_argument('--key', type=str, default='openai-key', help='Name of the HuggingFace API token variable name.')
    args = parser.parse_args()
    dataset = args.dataset
    
    #-------------------
    # load data
    #-------------------
    train_data = load_dataset(dataset, split='train')
    validation_data = load_dataset(dataset, split='validation')
    test_data = load_dataset(dataset, split='test')
    client = OpenAI(api_key=args.key)
    
    #--------------
    # inference
    #--------------
    if args.shottype == 'zero':
        print('Zero Shot...')
        zeroshot(test_data, client)
    elif args.shottype == 'one':
        print('One Shot...')
        oneshot(train_data, test_data, client)
    else:
        print('Few Shot...')
        fewshot(train_data, test_data, client)
    
if __name__ == "__main__":
    main()
