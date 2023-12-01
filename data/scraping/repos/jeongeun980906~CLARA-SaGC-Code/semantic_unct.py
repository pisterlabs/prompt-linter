from typing import Tuple
import os
import sys
import torch
import time
import json
from pathlib import Path
import spacy
import random
import numpy as np
import scipy
import math
import openai
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from llm.prompts import get_prompts
from llm.prompts import get_prompts, PROMPT_STARTER



class semantic_uncertainty():
    def __init__(self, task= 'cook'):
        self.few_shots = get_prompts(False, task)
        self.nlp = spacy.load('en_core_web_lg')
        self.new_lines = ""
        self.task = task
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
        self.model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").cuda()
        self.objects = ["blue block", "red block", "yellow bowl", "green block", "green bowl",'blue bowl']
        self.people = ["person with yellow shirt", "person with black shirt", "person with white shirt"]
        self.num_infer = 10
        self.people = []
        self.floor_plan = []
        self.inference = self.inference_gpt

    def set_prompt(self):
        des = PROMPT_STARTER[self.task]
        des += "Follow are examples"
        for c in self.few_shots:
            des += c
        des += "From this, predict the next action with considering the role of the robot and the ambiguity of the goal"
        if self.task =="clean":
            temp = ""
            for e, obj in enumerate(self.floor_plan):
                temp += obj
                if e != len(self.floor_plan)-1:
                    temp += ", "
            des += "objects = [" + temp + "] \n"
        
        if self.task == 'cook' or self.task =="clean":
            temp = ""
            for e, obj in enumerate(self.objects):
                temp += obj
                if e != len(self.objects)-1:
                    temp += ", "
            des += "objects = [" + temp + "] \n"
        
        if self.task == 'mas':
            temp2 = ""
            for e, obj in enumerate(self.people):
                temp2 += obj
                if e != len(self.people)-1:
                    temp2 += ", "
            des += "scene: people = [" + temp2+ "] \n"
        # des += "\n The order can be changed"
        des += "goal:{}\n".format(self.goal)
        if self.new_lines != "":
            des += self.new_lines
        self.prompt = des

    def inference_gpt(self):
        while True:
            try:
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=self.prompt,
                    temperature=0.5,
                    max_tokens=128,
                    top_p=1,
                    frequency_penalty=0.2,
                    presence_penalty=0,logprobs = 5, stop=")"
                )
                break
            except:
                time.sleep(1)
                continue
        logprobs = response['choices'][0]['logprobs']['token_logprobs']
        tokens = response['choices'][0]['logprobs']['tokens']
        top_logprobs = response['choices'][0]['logprobs']['top_logprobs']
        action_probs = []
        flag2 = False
        flag = False
        object = ""
        action = ""
        object_prob = 0
        object_probs = []
        object_num = 0
        action_num = 0
        action_prob = 0
        # print(tokens)
        for prob, tok, top_logprob in zip(logprobs[1:], tokens[1:],top_logprobs[1:]):
            if tok == ' done' or tok == 'done':
                return 'done', prob, 'done', prob
            if tok == ")" or tok == " )":
                flag = False
            if tok == "(":
                flag = True
                flag2  = False
                continue
            if tok == ".":
                flag2 = True
            elif flag and not flag2:
                if tok !="":
                    # print(top_logprob)
                    object_probs.append(top_logprob)
                    object_prob += prob
                    object += tok
                    object_num += 1
            elif flag2 and not flag:
                if tok !="":
                    # print(top_logprob)
                    action_probs.append(top_logprob)
                    action += tok
                    action_num += 1
                    action_prob += prob
            
            if tok == '\n':
                break
        if object_num != 0:
            object_prob /= object_num
        else:
            object_prob = 0
            object_probs = []
        if action_num == 0:
            action_prob =  0
        # print(object_prob, action_prob)
        return object,object_prob, action, action_prob
    def reset(self):
        self.new_lines = ""

    def set_goal(self, goal):
        self.goal = goal

    def plan_with_unct(self):
        self.set_prompt()
        obj_cand = []
        obj_probs = []
        subj_cand = []
        subj_probs = []
        for _ in range(self.num_infer):
            object,object_prob, subject,subject_prob = self.inference()
            if len(object) > 2:
                obj_cand.append(object)
                obj_probs.append(object_prob)
            if len(subject) > 2:
                subj_cand.append(subject)
                subj_probs.append(subject_prob)
        semantic = self.deberta(obj_cand, subj_cand)
        if len(obj_cand)>0:
            pick_ent = self.get_entropy(semantic, obj_cand, obj_probs, 1)
        else: pick_ent = 0
        if len(subj_cand)>0:
            place_ent = self.get_entropy(semantic, subj_cand, subj_probs, 0)
        else: place_ent = 0
        unct= {
            'obj':pick_ent,
            'ac':place_ent,
            'total':pick_ent+place_ent
        }
        tasks = []
        scores = []
        for x,y in zip(obj_cand, subj_cand):
            prompt = 'robot action: robot.{}({})'.format(y,x)
            if prompt not in tasks:
                tasks.append(prompt)
                scores.append(1)
            else:
                scores[tasks.index(prompt)] += 1

        return tasks, scores, unct

    def get_entropy(self,semantic, cands, probs, flag_index = 0):
        log_pobs = {}
        for value in semantic.values():
            log_pobs[value] = {}
        for x, log_pob in zip(cands, probs):
            for key in semantic.keys():
                if x in key:
                    break
            try:
                print(key)
            except:
                continue
            idx = semantic[key]
            try:
                log_pobs[idx][x].append(log_pob)
            except:
                log_pobs[idx][x] = [log_pob]
        unct = 0
        for indx,val in log_pobs.items():
            for key,x in val.items():
                x = sum(x)/len(x)
                val[key] = math.exp(x)
            sum_probs = sum(val.values())
            if sum_probs>1: sum_probs = 1
            if sum_probs <= 0: sum_probs = 1e-6
            unct += math.log(sum_probs)#*sum_probs # sum probability then take log
        if len(log_pobs) == 0:
            return 0
        unct = -unct/len(log_pobs) # average over number of semantic classes
        return unct

    def deberta(self, cands, cands2):
        total_cands = [c2.lower() + ' ' + c1.lower() for c1,c2 in zip(cands,cands2)]
        unique_generated_texts = list(set(total_cands))
        answer_list_1 = []
        answer_list_2 = []
        inputs = []
        semantic_set_ids = {}
        for index, answer in enumerate(unique_generated_texts):
            semantic_set_ids[answer] = index
        question = 'robot should'
        deberta_predictions = []
        if len(unique_generated_texts) > 1:
            # Evalauate semantic similarity
            for i, reference_answer in enumerate(unique_generated_texts):
                for j in range(i + 1, len(unique_generated_texts)):

                    answer_list_1.append(unique_generated_texts[i])
                    answer_list_2.append(unique_generated_texts[j])

                    qa_1 = question + ' ' + unique_generated_texts[i]
                    qa_2 = question + ' ' + unique_generated_texts[j]

                    input = qa_1 + ' [SEP] ' + qa_2
                    inputs.append(input)
                    encoded_input = self.tokenizer.encode(input, padding=True)
                    prediction = self.model(torch.tensor(torch.tensor([encoded_input]), device='cuda'))['logits']
                    predicted_label = torch.argmax(prediction, dim=1)

                    reverse_input = qa_2 + ' [SEP] ' + qa_1
                    encoded_reverse_input = self.tokenizer.encode(reverse_input, padding=True)
                    reverse_prediction = self.model(torch.tensor(torch.tensor([encoded_reverse_input]), device='cuda'))['logits']
                    reverse_predicted_label = torch.argmax(reverse_prediction, dim=1)

                    deberta_prediction = 1
                    # print(qa_1,'|', qa_2, predicted_label, reverse_predicted_label)
                    if 0 in predicted_label or 0 in reverse_predicted_label:
                        has_semantically_different_answers = True
                        deberta_prediction = 0

                    else:
                        semantic_set_ids[unique_generated_texts[j]] = semantic_set_ids[unique_generated_texts[i]]

                    deberta_predictions.append([unique_generated_texts[i], unique_generated_texts[j], deberta_prediction])
        # print(deberta_predictions)
        return semantic_set_ids
    
    def append(self, object, subject, task=None):
        if task == None:
            next_line = "\n" + "    robot.{}({})".format(subject,object)
        else:
            next_line = "    " + task +"\n"
        self.new_lines += next_line
