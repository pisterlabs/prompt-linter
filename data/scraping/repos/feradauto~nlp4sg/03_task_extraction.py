import argparse
import time
import os
import pandas as pd
import numpy as np
import openai
import time
import csv
from datasets import load_dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

def load_data(dataset):
    dataset = load_dataset("csv", data_files={"test": dataset})['test']
    return dataset

class MLModel:
    def __init__(self):
        pass
    
    def predict(self, data):
        
        prediction = self.perform_prediction(data)
        processed_response = self.process_response(prediction)
        return processed_response
    
    def perform_prediction(self, processed_data):
        raise NotImplementedError
    def process_response(self, data):
        raise NotImplementedError

class OpenAIModel(MLModel):
    def __init__(self,model_version=None):
        self.model_version = model_version
        self.preprompt="""Identify the NLP task(s) that this paper is dealing with. Select a text span that is an appropriate answer, or if no span serves as a good answer, just come up with a phrase. Examples of tasks are: fake news detection, name entity recognition, question answering, etc."""
        self.question="""The primary NLP task addressed in this paper is:"""
        self.OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
        self.response_columns=['prompt','response','task']
    def perform_prediction(self, data):
        prediction=dict()
        data=self.preprompt+"\n"+data+"\n"+self.question
        completion = openai.Completion.create(engine=self.model_version, prompt=data,temperature=0,max_tokens=100,logprobs=1)
        prediction['response']=completion.choices[0].text
        prediction['prompt']=data
        return prediction
    def process_response(self, predictions):

        predictions['task']=predictions['response'].replace("\n", "").rstrip(".").lstrip(' ')
        predictions['task']=predictions['task'].split(',')
        return predictions

class ZeroShotQA(MLModel):
    def __init__(self,model_version=None):
        self.model_version = model_version
        self.qa = pipeline("question-answering",
                          model=model_version,tokenizer=model_version, device=-1)
        self.response_columns=['answer','task']
    def perform_prediction(self, data):
        QA_input = {
        'question': "Which NLP task does this paper address?",
        'context': data
        }
        prediction = self.qa(QA_input)
        return prediction
    def process_response(self, predictions):
        predictions['task']=predictions['answer'].replace("\n", "").rstrip(".").lstrip(' ')
        predictions['task']=predictions['task'].split(',')
        return predictions

def main(args):
    data=load_data("./results_task_1.csv")
    if args['model'] in ['text-davinci-002']:
        model=OpenAIModel(args['model'])
    elif args['model'] in ['bert-large-uncased-whole-word-masking-finetuned-squad']:
        model=ZeroShotQA(args['model'])
    else:
        model=ZeroShotQA('bert-large-uncased-whole-word-masking-finetuned-squad')

    with open("task_extr_task3.csv", 'w', newline='') as file:
        csv_writer = csv.writer(file)
        cols=['ID', 'title', 'abstract', 'text', 'year', 'nlp4sg_score']
        csv_writer.writerow(cols+model.response_columns)
        for d in data:
            output=model.predict(d['text'])
            final_output=[d['ID'],d['title'],d['abstract'],d['text'],d['year'],d['nlp4sg_score']]
            for c in model.response_columns:
                final_output.append(output[c])
            csv_writer.writerow(final_output)

if __name__ == '__main__':
    args=argparse.ArgumentParser()
    ## openai if you want to use GPT 3
    args.add_argument("--model",type=str,default="bert-large-uncased-whole-word-masking-finetuned-squad")
    args=vars(args.parse_args())
    main(args)
