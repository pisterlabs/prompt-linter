import sys
import os
import json
from urllib import response
import matplotlib.pyplot as plt
import operator
import cohere
import random
from dotenv import load_dotenv
from difflib import SequenceMatcher

sys.path.append(os.path.abspath(os.path.join("./Scripts/")))
import config


class Prompter():
    def __init__(self, inputval, model='xlarge', prompt_type=1,num_tokens=150, example_num=2) -> None:
        self.data = {}
        with open('./Data/relations_train.json', 'r') as f:
            self.data = json.loads(f.read())
        self.inputval = inputval
        self.model = model
        self.num_tokens = num_tokens
        self.example_num = example_num
        self.prompt_type = prompt_type

        #Configure cohere api
        API_KEY = config.cohere_api['api_key']
        self.co = cohere.Client(API_KEY)


    def extract_values(self, data):
            #extract the values from each json dictionaries and convert them to a prompt types
        s: str = ''
        for d in data:
            s+='Job Description: '
            s+=str(d['document']).strip().replace('\n', ' ').strip()
            s+='\n'
            
                #Extract entities from tokens
            token_dict = {}
            for token in d['tokens']:
                entity_label = token['entityLabel']
                if entity_label in token_dict.keys():
                    token_dict[entity_label] += f",{token['text']}"
                else:
                    token_dict[entity_label] = f"{token['text']}"
            dict_str = ''
            for key,value in token_dict.items():
                dict_str+=f"{key}: {value}\n"
            s+=dict_str
            s+='\n--\n'
        return s

    def preprocess(self):
        self.inputval = str(self.inputval).strip().replace('/n', ' ')

    def prepare_prompt(self):
        p_val = self.prompt_type
        if p_val == 1:
            prompt_data = []
            similarity = {}
            for i, d in enumerate(self.data):
                similarity[i] = SequenceMatcher(None, self.inputval, d['document']).ratio()
            sorted_dict = dict( sorted(similarity.items(), key=operator.itemgetter(1),reverse=True))
            max_p = self.example_num
            i = 0
            for k,v in sorted_dict.items():
                prompt_data.append(self.data[k])
                i+=1
                if i>=max_p: break
            self.pr_data = self.extract_values(prompt_data)
        else:
            rand_items = random.sample(self.data, self.example_num)
            self.pr_data = self.extract_values(rand_items)
            
    def send_request(self):
        prompt = self.pr_data
        prompt+=f'Job Description: {self.inputval}'
       
        response = self.co.generate( 
            model=self.model, 
            prompt=prompt, 
            max_tokens=self.num_tokens, 
            temperature=0.5, 
            k=0, 
            p=1, 
            frequency_penalty=0, 
            presence_penalty=0, 
            stop_sequences=["--"], 
            return_likelihoods='NONE')
        return response.generations[0].text

    def post_process(self, response):
        response = str(response).strip().split('\n')
        response_dict = {}
        for vl in response:
            try:
                splitted_val = vl.split(':')
                response_dict[splitted_val[0]] = splitted_val[1]
            except Exception as e:
                pass

        return response_dict
        

    def make_request(self):
        self.preprocess()
        self.prepare_prompt()
        response = self.send_request()
        trial = 0
        max_tries = 3
        response = response.replace('--', '').strip()
        while len(response) < 10:
            trial+=1
            if trial > max_tries:
                break
            response = self.send_request()
            response = response.replace('--', '').strip()
        if len(response) < 10:
            raise Exception("Can not get a meaningful response")
        else:
            if self.model=='xlarge':
                return self.post_process(response), True
            return response, False

     #job desccriptions


    
    

    
    
    

    
        

