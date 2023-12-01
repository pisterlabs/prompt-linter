#Pokemon Showdown Voice Controller
#Integrated Dialogue Manager Functions
#Aditya Pandey
#Note: Check the README and report for a full understanding of how this works

from poke_env import PlayerConfiguration
from poke_env.player import Player, RandomPlayer
from difflib import SequenceMatcher
import asyncio
import time
import speech_recognition as sr
import keyboard
from fuzzywuzzy import fuzz
import fuzzy
from transformers import BertModel
import torch
import torch.nn as nn
from transformers import BertTokenizer
import numpy as np
import pandas as pd
import openai
import time
import os
from functools import partial
import random
from collections import Counter
import tqdm
import re
import pyautogui


#################################################################
#Part1: Phonemes
def get_phonetic_similarity(text1, text2):
    phonetic1 = fuzzy.nysiis(text1)
    phonetic2 = fuzzy.nysiis(text2)
    similarity = fuzz.ratio(phonetic1, phonetic2)
    return similarity

def mostsimilarphone(battle,obj,intent,threshold=50):

	moves=battle.available_moves
	switches=battle.available_switches

	max_score=0
	max_move=None
	max_str=''

	sets={'attack':moves,'switch':switches}

	#Chose the right set to evaluate

	chosen_set=sets[intent]

	for i in range(0,len(chosen_set)):

		if intent=='attack':
			this_move=str(chosen_set[i].id)
		else:
			this_move=str(chosen_set[i].species)

		score=get_phonetic_similarity(obj,this_move)

		if score>max_score:
			max_score=score
			max_move=chosen_set[i]
			max_str=this_move

	if max_score>threshold:
		print('Move Chosen: ',max_str,'Score: ',max_score)
		return max_move

	else:
		print('No move found')
		return False

#################################################################
#Part2: GPT Prompting

#few shot examples for open ai request
fs_test = [
    {"prompt": "Pikachu, use fireball","label": "(attack,fireball)"},
    {"prompt":"Charizard, come back. Go Squirtle!","label":"(switch,Squirtle)"},
    {"prompt":"articuno, hit him with blizzard storm","label":"(attack,blizzard storm)"},
    {"prompt":"use quickattack","label":"(attack,quickattack)"}
]

def pred_gpt(
        test_prompt,
        few_shot_examples=fs_test,
        model_name = "gpt-3.5-turbo",
        max_tokens = 50,
        temperature = 0.0,
):

	with open('key.txt') as f:
	    openai.api_key = f.read()

	    messages_prompt = [
	        {"role": "user", "content": """You are an expert in Pokemon names and moves.
	        You are given a text command made by a pokemon trainer in battle, and need to parse
	         it into its intent, its subject, and its object. There are only two types of intents: either 'attack'
	         or 'switch'. An attack refers to commanding the pokemon to attack. A switch refers to
	        switching out the current pokemon for another pokemon. The object will correspondingly be the
	        move the user wants the pokemon to use, or the name of the pokemon the user will switch to.
	         Give respones in the form (intent, object).
	        If there is no clear subject, then the subject is the current_pokemon."""}
	        ]

	    #Lets get the few shot examples in
	    for each in few_shot_examples:
	      prompt={"role":"user","content":each["prompt"]}
	      answer={"role":"assistant","content":each["label"]}
	      messages_prompt.append(prompt)
	      messages_prompt.append(answer)

	    #Now lets get our test prompt in
	    test={"role":"user","content":test_prompt}
	    messages_prompt.append(test)

	    model_output = None

	    while True:
	        try:
	            response=openai.ChatCompletion.create(
	              model="gpt-3.5-turbo",
	              messages=messages_prompt,
	              max_tokens=max_tokens,
	              temperature=temperature,
	            )
	            model_output=response['choices'][0]['message']['content']
	            time.sleep(1) # to prevent rate limit error
	            break
	        except (openai.error.APIConnectionError, openai.error.RateLimitError, openai.error.Timeout, openai.error.ServiceUnavailableError) as e:
	            #Sleep and try again
	            print(f"Couldn't get response due to {e}. Trying again!")
	            time.sleep(20)
	            continue

	    return model_output


#################################################################
#Part 3: Intent Prediction using a Bert Binary Classifier Model


#Main Model that we only want to initialize once throughout the lifetime of the game: 
model = BertBinaryClassifier()
model_path = "models/pokemodel.pt" 
model.load_state_dict(torch.load(model_path))

def predict_text(text, model=model, tokenizer=BertTokenizer.from_pretrained("bert-base-uncased"),device = "cpu"):

    model = model.to(device)
    model.eval()

    pred_label = None
    tokenized_input=tokenizer(text,return_tensors="pt")
    input_ids=tokenized_input["input_ids"]
    attn_mask=tokenized_input["attention_mask"]

    input_ids=input_ids.to(device)
    attn_mask=attn_mask.to(device)

    # Send the input_ids_dict and attn_mask_dict to device


    # Step 2: Feed the input_ids_dict and attn_mask_dict to the model and get the final predictions
    # (Don't forget torch.no_grad())

    with torch.no_grad():
      pred = model(input_ids,attn_mask).cpu().detach().numpy()[0,0]
      pred_value=np.round(pred,0)

      pred_dict={0:"attack",1:"switch"}

    pred_label_hr = pred_dict[pred_value]

    return pred_label_hr