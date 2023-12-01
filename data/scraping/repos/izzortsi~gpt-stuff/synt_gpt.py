#%%


import openai
import os
import time
import sys
import json
from dataclasses import dataclass
from typing import List
from text_generation import generate, complete

# openai.api_key = os.environ.get("OPEN_AI_API_KEY")
# openai.api_base = 'https://api.openai.com/v1'
# MODEL = "gpt-3.5-turbo-0613"

openai.api_key = os.environ.get("OPEN_AI_FREE_API_KEY")
openai.api_base = 'https://api.pawan.krd/v1'

MODEL = "gpt-3.5-turbo"
#You will have to organize all this information and help me make sense of it.
SYS_PROMPT = """You are a smart text expander. Your goal is to expand poorly given instructions for tasks into a list of instructions that achieves the objective implied by the poorly given instructions.
                I will give you some examples of the instructions that you will be given bellow, delimited by backticks. Remember them:
                
                `so tasks

                1. run that script and check output (it doesnt have to run to completion)

                2. modify stable diffusion library so we can feed in noise vector by hand

                3. look at the model.py thing
                - see if we can make a file to load the unet, the text encoder and the auto-encoder as seperate/independent classes
                - so we can load one and use only one without running whole model`

                `1. revert changes
                 2. make a new class
                 - one class, in one file, for each of the models
                 - encode/decode for auto-encoder, etc
 
                 3. function for load, unload
 
                 4. function for batched vs single input
 
                 5. In Scripts
                 - make a new example
                 - where you load the 3 classes
                 - then encode the prompt
                 - then run denoising 20 times
                 - then output latent to image
                 -- function to save image
 
                 6. Make sure the inner loop, we can feed in noise vector directly
 
                 But dont change the old pipeline files yet`

                `in the main /existing work flow; you use all three
                - unet
                - denoiser
                - clip

                in one run

                But in the alternative work flow

                - you load clip
                -- run 1024 times
                - unload clip

                - you load denoiser
                -- you run 1024 times
                - unload denoise

                - you load encoder
                -- you run 1024 times
                - unload encoder


                So its "Batched" so you are not switching between which network is used during runtime`


                `Ticket:

                1. copy the files over
                - duplicate it

                2. Make function
                - loads whole model with first copy

                3. Make function
                - saves 3 models
                -- embedding
                -- unet
                -- variational auto encoder

                as seperate files to output

                4. Load those files with second model
                - and track the tile
                - each model is a seperate file
                -- model_clip.y
                -- model_unet.py
                -- model_vae.py

                Each model class must have a Load and Unload function. No loading model on init.

                5. Use safe tensor, not .cpt; it might be faster and is the newer format

                6. Do test to make sure we are getting same result`

                Expand each of the instructions I gave as an example. Remember them as a list with objects of the format: <poorly given instructions>: <expansion>
                Call this list the database.
                Using the database as a reference, expand the instructions I will give you in future messages. Put them in the database, with the format: <poorly given instructions>: <expansion>
                
                If a message starts with @, it is a command. 
                If a message is not a command, it will be a poorly given instruction for a task, delimited by backticks.
                The following commands are available:
                @show: show me the database, in the format: <poorly given instructions>: <expansion>
                @retrieve <word>: retrieve all elements from the database that contain <word>. Show them in the format: <poorly given instructions>: <expansion>.
                """


TEMPERATURE = 0.5

class GPT:
    def __init__(self, sys_prompt=SYS_PROMPT, model=MODEL, temperature = TEMPERATURE):
        self._sys_messages = [{"role": "system", "content": sys_prompt}]
        self._messages = self._sys_messages
        self.response = ""
        self._model = model
        self._temperature = temperature

    def set_system(self, sys_prompt):
        self._sys_messages = [{"role": "system", "content": sys_prompt}]
    
    def add_system(self, sys_prompt):
        self._sys_messages.append({"role": "system", "content": sys_prompt})

    def completion(self, prompt, role = "user", chat=False):
        user_message = [{"role": role, "content": prompt}]
        self._messages += user_message
        response = openai.ChatCompletion.create(
            model=self._model,
            messages=self._messages,
            temperature=self._temperature, # this is the degree of randomness of the model's output
            max_tokens=1000,
        )
        self.response = response.choices[0].message["content"]
        self._messages += [{"role": "assistant", "content": self.response}]
            
        return self.response

def chat(gpt):
    while True:
        prompt = input("You: ")
        if prompt == "exit":
            break

        print("Bot:", gpt.completion(prompt, chat=True))

GPT.chat = chat
#%%


if __name__ == "__main__":
    gpt = GPT()
    if len(sys.argv) > 1:
        gpt.chat()

# %%
# gpt = GPT()
# #%%
# gpt.completion("I have to do the dishes", role="user")
# %%
