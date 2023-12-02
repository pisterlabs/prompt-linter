import os
import json
import glob
import re
from multiprocessing import Pool
from tqdm import tqdm
import torch
import openai
import itertools
import random
#import environment
import tiktoken



####### HELPER FUNCTIONS ##########

#This class stores the history of the conversation which is used as context
class MemoryList(list): #Sliding window implementation for now
    def __init__(self, *args, max_tokens = 3500):
        super().__init__(*args)
        self.max_tokens = max_tokens
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        
    #Add smth to the list -> remove first item until total tokens < 4000
    def append(self, item):
        print(item)
        total_tokens = self.check_tokens()
        item2 = item["content"]
        while len(self.tokenizer.encode(item2)) + total_tokens > self.max_tokens:
            if len(self.tokenizer.encode(item2)) > self.max_tokens:
                self.summarize()
                raise Exception("Item too long")
            self.handle_overflow()
            total_tokens = self.check_tokens()
        super().append(item)

    #Helper to check no of tokens
    def check_tokens(self):
        return sum(len(self.tokenizer.encode(item['content'])) for item in self)
    
    #Helper to handle overflow
    def handle_overflow(self):
        if len(self) >0:
            self.pop(0)

    #For now it will just be a signal to terminate the generation
    def summarize(self):
        print("Summarizing") #End goal is to use gpt-16k to do this


#This is a helper function to convert skills retrieved from the vector DB
#into the relevant code
def search_files(skills,run):
    data_dir = f"chainofaction/data/run_{run}"
    all_code = []
    for skill in skills:
        print(skill)
        with open(os.path.join(data_dir,os.path.splitext(skill)[0]+'.py')) as f:
            lines = f.readlines()
            res = ""
            for i,line in enumerate(lines):
                res+= "\n"+line.strip() if not i else line.strip()
        all_code.append(res)
    return all_code
     
        

#This code handles the generation API calls
def generate(messages, max_tokens = 2048, temperature = 0.0, model = "gpt-4"):
    if  model in ["gpt-4", "gpt-4"]:
        params = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages
        }
        for retry in range(3):
            try:
                return openai.ChatCompletion.create(**params)["choices"][0]["message"]["content"]
            except Exception as e:
                return e
    
    # For older models, use the completion API with max_tokens=1024
    params = {
        "model": model,
        "max_tokens": min(max_tokens, 1024),
        "temperature": temperature,
        "prompt": messages[-1]
    }
    for retry in range(3):
        try:
            return openai.Completion.create(**params)["choices"][0]["text"]
        except:
            pass



########### AGENT CLASS ############
class ZeroShotAgent:
    def __init__(self,db,environment, model = "gpt-4", max_tokens = 2048, temperature = 0.0, explore_ratio = 0.3, max_count = 0):
        '''
        Agent: A class that handles the generation of skills and the interaction with the environment
        model: the model to use for generation
        max_tokens: the maximum number of tokens to generate
        temperature: the temperature to use for generation
        explore_ratio: the ratio of times to explore (deprecated for now)
        max_count: the maximum number of times to generate (deprecated for now)
        db: the vector database to use
        '''
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.message = MemoryList(max_tokens = 3500)
        self.db = db
        self.explore_ratio = explore_ratio 
        self.max_count = max_count  
        self.environment = environment     

    def get_prompt(self,task): #Helper function to get the prompt
        with open(f"chainofaction/prompts/{task}") as f:
            return f.read().strip()

    def decompose(self,problem): #Helper function to decompose the problem into steps
        decompose = self.get_prompt("decompose.txt")
        decompose += "\n\n" + problem
        self.message.append({"role": "user", "content": decompose})

        for retry in range(3): #Loop to retry if generation fails
            try:
                skills = generate(self.message, max_tokens = 2048, temperature = 0.0, model = "gpt-4")
                print(skills)
                self.message.append({"role": "assistant", "content": skills})
                return skills
            except Exception as e: #Add error for model to debug
                print("Error: Failed to generate response")
                self.message.append({"role":"user","content": f"Failed to execute solution generation: {e}"})
    
    
       
    
    def zeroshot_soln(self, problem, steps,fn_head):
        try:
            self.message.append({"role": "user", "content": f"Here is the problem: {problem}\nHere are the identified steps: {steps}\nWrite Python code to solve the problem\n Use this function head:{fn_head}"})
            soln = generate(self.message)
            self.message.append({"role":"assistant","content": f"{soln}"})
            return soln
        except Exception as e:
            print("ERROR ZERO SHOT")
            raise e
            return None
   

    def get_response(self,problem, cases, fn_head, title):
        '''
        problem: description of problem
        '''
        success = True

        steps = self.decompose(problem)

        soln = self.zeroshot_soln(problem, steps, fn_head)
        
        if not soln:
            return None
    

        ###Iterative prompting
        for retry in range(10):
            #TODO
            #Check if code can get correct answer for test cases
            #If not, prompt for more code
            try:
                passed = True #Dummy variables
                output = ''
                output, passed = self.environment.execute(soln, cases) #Environment implemented in env.py later

                    
            except Exception as e:
                print(e)
                raise e
                print("Error: Failed to generate response")
                self.message.append({"role":"user","content": f"Failed to execute iterative prompting: {type(e)}"})

        if not passed:
            success = False

        #save into chroma before return statement
        if success:
            self.message.append({"role": "user", "content": f"Write a description of what this program solves:\n{soln}"})
            desc = generate(self.message)
            with open(f"chainofaction/data/run_{self.environment.run}/newdesc/"+title,'w') as f:
                f.write(soln)
            return (soln, desc, title)
        return None
    
    def reset(self):
        self.message = MemoryList(max_tokens = 3500)
