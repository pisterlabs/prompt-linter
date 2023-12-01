import json
import argparse
import pandas as pd
import numpy as np 
import random
from sklearn.metrics import f1_score, accuracy_score
from evaluate import load
squad_metric = load("squad_v2")

import os
import requests
import json
import openai
from sklearn.utils import shuffle

openai.api_key = ""
openai.api_base =  ""
openai.api_type = ""
openai.api_version = ""
deployment_id=""

def conversion(prediction_str: str) -> str:
    mapper = {'N/A': ''}    
    if prediction_str in mapper:
        return mapper[prediction_str]
    else:
        return prediction_str

def precision_filter(pred: str, context: str) -> str:
    mapper = {'N/A': ''}
    if pred in mapper:
        return mapper[pred]
    elif not pred in context: 
        return ''
    else:
        return pred

def compute_metrics_prompt(context, predictions, answers):
    """Given some predictions, calculate the F1. """
    # Decode the predictions + labels 
    clean_predictions = [{'prediction_text': conversion(x), 'id': str(i), 'no_answer_probability': -1} for i, x in enumerate(predictions)]
    precision = [{'prediction_text': precision_filter(x, context[i]), 'id': str(i), 'no_answer_probability': -1} for i, x in enumerate(predictions)]

    references = [{'id': str(i), 'answers': l} for i, l in enumerate(answers)]

    results = squad_metric.compute(predictions=clean_predictions, references=references)
    clean_results = squad_metric.compute(predictions=precision, references=references)
    return {'reg': results, 'precision': clean_results}
 
def format_single_example(e, prompt: str) -> str:
    if prompt == '0':
        return f"Context: {e['context']}\nQuestion: {e['question']}\nAnswer:"
    elif prompt == '1':
        return f"Context: {e['context']}\nGiven the above context, {e['question']}? Answer N/A if there is no answer."
    elif prompt == '2':
        return f"Answer N/A if there is no answer.\n\nContext: {e['context']}\nQuestion:{e['question']}\nAnswer:"
    elif prompt == '3':
        return f"Context: {e['context']}\nGiven the above context, {e['question']}? Answer N/A if there is no answer or give a quote from the context: "
    elif prompt == '4':
        return f"Context: {e['context']}\nGiven the above context, {e['question']}?\nAnswer: \""
    elif prompt == '5':
        return f"Context: {e['context']}\nGiven the above context, {e['question']}? Give a quote from the text: "
    elif prompt == '6':
        prompt = "Context: IMPRESSION:  Subdural hematomas with blood products of different ages.\n Question vescular abnormality in left suprasellar space.  Findings were\n discussed with Dr. [**Last Name (STitle) 8620**] at 9:25 am on [**2191-8-5**].  An MRI of the brain and MRA\n of the COW is recommended.\nGiven the above context, Is there any significant change in bleeding? Answer N/A if there is no answer or give a quote from the context: N/A\n\n"
        return prompt + f"Context: {e['context']}\nGiven the above context, {e['question']}? Answer N/A if there is no answer or give a quote from the context: "
    elif prompt == '7':
        prompt = "Context: FINAL REPORT\nCHEST SINGLE AP FILM:\n\nHISTORY: ICD placement and shortness of breath.\n\nThere is a right sided dual chamber ICD with leads unchanged in location in\n this single view compared with the prior film of [**2101-5-25**]. No pneumothorax.\n There is cardiomegaly with upper zone redistribution, bilateral pleural\n effusions and associated bibasilar atelectases. Consolidation at the lung\n bases cannot be ruled out.\nGiven the above context, Is there any significant change from prior visit? Answer N/A if there is no answer or give a quote from the context: ICD with leads unchanged in location\n\n"
        prompt += f"Context: {e['context']}\nGiven the above context, {e['question']}? Answer N/A if there is no answer or give a quote from the context: "
        return prompt 
    elif prompt == '8':
        prompt = "Context: IMPRESSION:  Subdural hematomas with blood products of different ages.\n Question vescular abnormality in left suprasellar space.  Findings were\n discussed with Dr. [**Last Name (STitle) 8620**] at 9:25 am on [**2191-8-5**].  An MRI of the brain and MRA\n of the COW is recommended.\nGiven the above context, Is there any significant change in bleeding? Answer N/A if there is no answer or give a quote from the context: N/A\n\nContext: FINAL REPORT\nCHEST SINGLE AP FILM:\n\nHISTORY: ICD placement and shortness of breath.\n\nThere is a right sided dual chamber ICD with leads unchanged in location in\n this single view compared with the prior film of [**2101-5-25**]. No pneumothorax.\n There is cardiomegaly with upper zone redistribution, bilateral pleural\n effusions and associated bibasilar atelectases. Consolidation at the lung\n bases cannot be ruled out.\nGiven the above context, Is there any significant change from prior visit? Answer N/A if there is no answer or give a quote from the context: ICD with leads unchanged in location\n\n"
        prompt += f"Context: {e['context']}\nGiven the above context, {e['question']}? Answer N/A if there is no answer or give a quote from the context: "
        return prompt
    elif prompt == '9':
        prompt = "Context: IMPRESSION:  Subdural hematomas with blood products of different ages.\n Question vescular abnormality in left suprasellar space.  Findings were\n discussed with Dr. [**Last Name (STitle) 8620**] at 9:25 am on [**2191-8-5**].  An MRI of the brain and MRA\n of the COW is recommended.\nGiven the above context, Is there any significant change in bleeding? Answer N/A if there is no answer or give a quote from the context: N/A\n\n"
        prompt += "Context: FINAL REPORT\nCHEST SINGLE AP FILM:\n\nHISTORY: ICD placement and shortness of breath.\n\nThere is a right sided dual chamber ICD with leads unchanged in location in\n this single view compared with the prior film of [**2101-5-25**]. No pneumothorax.\n There is cardiomegaly with upper zone redistribution, bilateral pleural\n effusions and associated bibasilar atelectases. Consolidation at the lung\n bases cannot be ruled out.\nGiven the above context, Is there any significant change from prior visit? Answer N/A if there is no answer or give a quote from the context: ICD with leads unchanged in location\n\n"
        prompt += f"Context: {e['context']}\nGiven the above context, {e['question']}? Answer N/A if there is no answer or give a quote from the context: "
        return prompt
    elif prompt == '10':
        prompt = f"Context: {e['context']}\nGiven the above context, {e['question']}?\nGive an answer from the text, but do not reply if there is no answer: "
        return prompt
    elif prompt == '11':
        if e['question'][-1] == '?':
            e['question'] = e['question'][:-1]

        return f"Radiology Report: {e['context']}\nGiven the above radiology report, {e['question']}? Answer N/A if there is no answer or give a quote from the context: "
    elif prompt == '12':
        begin = "Find the answer to the question based on the context below. Provided a quote from the context, otherwise write N/A.\n\n"
        begin += f"Context: {e['context']}\nGiven the above context, {e['question']} Answer:"
        return begin
    elif prompt == '13':
        begin = "Find the answer to the question based on the context below. Provided a quote from the context, otherwise write N/A.\n\n"
        begin += f"Context: {e['context']}\nQuestion: {e['question']} Answer:"
        return begin
    elif prompt == '14':
        begin = "Find the answer to the question based on the context below. Provided a quote from the context, otherwise write N/A.\n\n"
        begin += "Context: IMPRESSION:  Subdural hematomas with blood products of different ages.\n Question vescular abnormality in left suprasellar space.  Findings were\n discussed with Dr. [**Last Name (STitle) 8620**] at 9:25 am on [**2191-8-5**].  An MRI of the brain and MRA\n of the COW is recommended. Is there any significant change in bleeding? Answer: N/A\n\n"
        begin += "Context: FINAL REPORT\nCHEST SINGLE AP FILM:\n\nHISTORY: ICD placement and shortness of breath.\n\nThere is a right sided dual chamber ICD with leads unchanged in location in\n this single view compared with the prior film of [**2101-5-25**]. No pneumothorax.\n There is cardiomegaly with upper zone redistribution, bilateral pleural\n effusions and associated bibasilar atelectases. Consolidation at the lung\n bases cannot be ruled out.\nQuestion: Is there any significant change from prior visit? Answer: ICD with leads unchanged in location\n\n"
        begin += f"Context: {e['context']}\nQuestion: {e['question']} Answer:"
        return begin
    elif prompt == '15':
        begin = "Find the answer to the question based on the context below. Provided a quote from the context, otherwise write N/A.\n\n"
        begin += "Context: IMPRESSION:  Subdural hematomas with blood products of different ages.\n Question vescular abnormality in left suprasellar space.  Findings were\n discussed with Dr. [**Last Name (STitle) 8620**] at 9:25 am on [**2191-8-5**].  An MRI of the brain and MRA\n of the COW is recommended.\nQuestion: Is there any significant change in bleeding?\nAnswer: N/A\n\n"
        begin += "Context: FINAL REPORT\nCHEST SINGLE AP FILM:\n\nHISTORY: ICD placement and shortness of breath.\n\nThere is a right sided dual chamber ICD with leads unchanged in location in\n this single view compared with the prior film of [**2101-5-25**]. No pneumothorax.\n There is cardiomegaly with upper zone redistribution, bilateral pleural\n effusions and associated bibasilar atelectases. Consolidation at the lung\n bases cannot be ruled out.\nQuestion: Is there any significant change from prior visit?\nAnswer: ICD with leads unchanged in location\n\n"
        begin += 'Context: IMPRESSION:  1)  Large constricting circumferential mass is present in the\n antrum of the stomach, causing massive dilatation of the body and fundus of\n the stomach.  Adenopathy is present in the gastrohepatic ligament.  No\n definite metastatic lesions are seen in the liver, although there is a tiny\n low attenuation lesion in the posterior segment of the right hepatic lobe\n which is too small to characterize.  No other evidence of metastatic disease\n is seen in the torso.  2)  Markedly enlarged prostate.  3)  Bilateral small\n anastomotic aneurysms at the junction of the bypass graft with the common\n femoral arteries.  4)  Small fat-containing left inguinal hernia.\nQuestion: Did the gastric cancer metastasize to chest.\nAnswer: No other evidence of metastatic disease\n is seen in the torso\n\n'
        begin += f"Context: {e['context']}\nQuestion: {e['question']}\nAnswer:"
        return begin
    else:
        raise ValueError("Prompt id not implemented")


def preprocess_function(examples, prompt: str):
    """Format the examples and then tokenize them. """
    inputs = [format_single_example(e, prompt) for _, e in examples.iterrows()]

    return {'inputs': inputs}

def format_csv(df: pd.DataFrame):
    df['answers'] = [eval(x) for x in df['answers']] #type: ignore
    return df

def get_data(radqa_path: str):
    """Get the radqa data. """
    train = format_csv(pd.read_csv(radqa_path + '/train.csv'))
    val = format_csv(pd.read_csv(radqa_path + '/dev.csv'))
    val = shuffle(val)

    test = format_csv(pd.read_csv(radqa_path + '/test.csv'))
    return {"train": train, "val": val, "test": test}

def get_prompted_data(dataset_dict, prompt: str):
    """Tokenize stuff. """
    processed_dict = {}
    for name, dataset_ in dataset_dict.items():
        processed_item = preprocess_function(dataset_, prompt)
        processed_item['answers'] = dataset_['answers'].values
        processed_item['context'] = dataset_['context'].values
        processed_dict[name] = processed_item
    
    return processed_dict

def query_openai_all(all_inputs):
    all_answers = []
    for i in range(0, len(all_inputs), 20):
        response = openai.Completion.create(engine=deployment_id, prompt=all_inputs[i:i+20], max_tokens=256, temperature=0)
        all_answers.extend([resp['text'].strip() for resp in response['choices']])

    return all_answers


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--radqa-dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--set', required=True, choices=['dev', 'test'])
    args = parser.parse_args()
    print(f"Running with {args.seed}")
 
    # Get data and use the tokenizer on the data 
    dataset_dict = get_data(args.radqa_dir)
    prompted_dataset = get_prompted_data(dataset_dict, args.prompt)
    
    if args.set == 'dev':
        ds = prompted_dataset['val']
        selection = 200
        ds = {'inputs': ds['inputs'][:selection], 'answers': ds['answers'][:selection], 'context': ds['context'][:selection]}
    else:
        ds = prompted_dataset['test']
    
    predictions = query_openai_all(ds['inputs'])
    output_metrics = compute_metrics_prompt(ds['context'], predictions, ds['answers'])
    print(output_metrics)