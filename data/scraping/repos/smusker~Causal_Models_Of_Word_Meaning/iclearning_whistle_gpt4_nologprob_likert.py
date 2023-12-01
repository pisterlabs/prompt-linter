import os
import openai
import random
import copy
import json
import numpy as np
import pandas as pd
from tabulate import tabulate
import re

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

openai.api_key = ""

kwargs={"temperature":0.7,"max_tokens":5,"top_p":1,"frequency_penalty":0,"presence_penalty":0}

samples_per_q = 5

prompt=""

randomize=True

naming=True
causal=False
function=False

ncf_num=-1

if(naming): 
    ncf_num=0
if(causal): 
    ncf_num=1
if(function): 
    ncf_num=2

whistle_outcome="called his dog"

hanger_outcome="hanged his T shirt"

gfork_outcome="loosened the garden soil"

def question_generator(ncf_num, object_name, outcome): 
    if (ncf_num==0):
        return ("Is it appropriate to call this object a "+object_name+"?")
    if (ncf_num==1):
        return("Is it likely that, as a result of the events described above, John "+outcome+"?")
    if (ncf_num==2):
        return("Does this scenario illustrate the function of a "+object_name+"?") 
    
@retry(wait=wait_random_exponential(min=5, max=25), stop=stop_after_attempt(80)) 
def chat_completion_with_backoff(prompt, **kwargs):


    raw_response = openai.ChatCompletion.create(
        model="gpt-4-0314",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            }, #"You are a helpful assistant with an excellent understanding of the physical world."},
            {"role": "user", "content": prompt},
        ],
        **kwargs
    )

    return raw_response

prompt+="""In this study we are studying knowledge of daily life activities. On the next pages you will find several short stories or scenarios, describing events in the daily lives of two characters. Each scenario is different from the others.
Read each one carefully. As you read them, imagine to yourself what is happening. In each scenario, there are two characters. The names of the two characters in all the scenarios are Jane and John. Jane has a young child, called John, who is constantly running around the house and trying new things.
Please use a 7 point rating scale in order to answer the first question that follows a scenario. A 1 always means “not at all”, a 4 means “somewhat”, and a 7 means “very well.”
Here are three examples that will help you to get used to this procedure.
First example:
Jane was reading a book that she was very interested in, and remembered she had to make a phone call. While she made the phone call, she left her book on the table. While Jane was on the phone, John looked for something to do. He saw the book that Jane was reading, picked it up, and browsed through it.
Question:
Does this scenario illustrate a situation where Jane is going to be mad at John?
Answer (respond with only a number between 1 and 7 inclusive, with 1 being "not at all" and 7 being "very well"): 1
Here is a second scenario:
On a certain occasion Jane was feeling stressed, extremely tired, and grouchy. To get her mind off her worries, she decided to watch some TV. While Jane was watching TV, John started to complain that he was hungry. Reluctantly, Jane had to get up and fix him something to eat.
Question:
Does this scenario illustrate a situation where Jane is going to be mad at John?
Answer (respond with only a number between 1 and 7 inclusive, with 1 being "not at all" and 7 being "very well"): 2
Here is a third scenario:
One day, Jane was getting ready to go out. She had decided to wear a nice outfit that she had recently bought. She left the outfit over her bed, and took a shower. While Jane was taking a shower, John came into the room carrying a permanent marker he found elsewhere. He decided to climb onto the bed, and in doing so he stained Jane’s outfit.
Question:
Does this scenario illustrate a situation where Jane is going to be mad at John?
Answer (respond with only a number between 1 and 7 inclusive, with 1 being "not at all" and 7 being "very well"): 7
The three examples you just read are about emotions. In contrast, the scenarios that you will read and rate next are not about emotions. However, they have the same structure."""

prompt+="""
One day Jane wanted to loosen the soil in her garden pots, but she didn’t have anything to do it with. So she decided to make something. She gathered all the materials and made it. When she finished, she left it in the garden so she could use it later. The object had three prongs and a handle. Later that day, John was looking for something to loosen the soil in the garden. He saw the object that Jane had made and thought that it would be good for loosening the soil in the garden. He grabbed the object by the handle and repeatedly pushed the prongs into the garden soil.
Question:
"""
prompt+=question_generator(ncf_num,"gardening fork",gfork_outcome)
prompt+="""
Answer (respond with only a number between 1 and 7 inclusive, with 1 being "not at all" and 7 being "very well"): 6
One day Jane was working in her metal shop, and she decided to make something. She gathered all the materials and started to work. As she worked, metal scraps were left to the side of her table. One of the scrap pieces looked like a spherical piece of metal with a handle attached to it. Later that day, John was looking for something play with. He saw the object that Jane had made and thought that it would be good for playing with. He grabbed the object from the spherical part, and repeatedly waved it above his head.
Question:
"""
prompt+=question_generator(ncf_num,"gardening fork",gfork_outcome)
prompt+="""
Answer (respond with only a number between 1 and 7 inclusive, with 1 being "not at all" and 7 being "very well"): 1
One day Jane was working in her metal shop, and she decided to make something. She gathered all the materials and started to work. As she worked, metal scraps were left to the side of her table. One of the scrap pieces looked like a wire twisted in different ways. Later that day, John was looking for something play with. He saw the object that Jane had made and thought that it would be good for playing with. He grabbed the object from both sides, and repeatedly squeezed it with his hands.
Question:
"""
prompt+=question_generator(ncf_num,"hanger",hanger_outcome)
prompt+="""
Answer (respond with only a number between 1 and 7 inclusive, with 1 being "not at all" and 7 being "very well"): 2
One day Jane wanted to hang her clothes, but she didn’t have anything to do it with. So she decided to make something. She gathered all the materials and made it. When she finished, she left it in her room so she could use it later. The object was a long wire shaped like the outline of a person's shoulders, and with a hook on the top. Later that day, John was looking for something to hang his clothes on. He saw the object that Jane had made and thought that it would be good for hanging his clothes on. He grabbed the object and fit it inside his T shirt so that the hook came out through the neck.
Question:
"""
prompt+=question_generator(ncf_num,"hanger",hanger_outcome)
prompt+="""
Answer (respond with only a number between 1 and 7 inclusive, with 1 being "not at all" and 7 being "very well"): 7
"""

#^yes no no yes

#experiment section starts here

c_none_p="One day Jane wanted to call her dog (who was out in the garden and was trained to answer to a high-pitch sound), but she didn’t have anything to do it with. So she decided to make something. She looked around the house for things that would allow her to make an object for calling her dog. She gathered all the materials and made it. When she finished, she left it on a table so she could use it later. The object was a conical sea shell that now had its tip broken. Later that day, John was looking for something to call his dog with. He saw the object that Jane had made and thought that it would be good for calling his dog. He grabbed the object, put its tip in his mouth, and blew." #one
c_hist_p="One day Jane wanted to clean up her desk. She reviewed different documents and objects that were on her desk and began to put all unwanted items in a cardboard box. Because she wasn’t careful when throwing objects into the box, the tip of one of the objects she discarded broke. The object was a conical sea shell that now had its tip broken. Later that day, John was looking for something to call his dog with. He saw the object that Jane had made and thought that it would be good for calling his dog. He grabbed the object, put its tip in his mouth, and blew." #two
c_struc_p="One day Jane wanted to call her dog (who was out in the garden and was trained to answer to a high-pitch sound), but she didn’t have anything to do it with. So she decided to make something. She looked around the house for things that would allow her to make an object for calling her dog. She gathered all the materials and made it. When she finished, she left it on a table so she could use it later. The object was a conical sea shell that now had its tip broken and replaced with a solid piece of plastic resin that completely blocked the opening. Later that day, John was looking for something to call his dog with. He saw the object that Jane had made and thought that it would be good for calling his dog. He grabbed the object, put its tip in his mouth, and blew." #three
c_goal_p="One day Jane wanted to call her dog (who was out in the garden and was trained to answer to a high-pitch sound), but she didn’t have anything to do it with. So she decided to make something. She looked around the house for things that would allow her to make an object for calling her dog. She gathered all the materials and made it. When she finished, she left it on a table so she could use it later. The object was a conical sea shell that now had its tip broken. Later that day, John was searching on the table for something to play with. He was distracted as he looked for something and inadvertently grabbed the sea shell. He grabbed the object, put its tip in his mouth, and blew." #four
c_act_p="One day Jane wanted to call her dog (who was out in the garden and was trained to answer to a high-pitch sound), but she didn’t have anything to do it with. So she decided to make something. She looked around the house for things that would allow her to make an object for calling her dog. She gathered all the materials and made it. When she finished, she left it on a table so she could use it later. The object was a conical sea shell that now had its tip broken. Later that day, John was looking for something to call his dog with. He saw the object that Jane had made and thought that it would be good for calling his dog. He grabbed the object, put his mouth near the wider opening, and whispered his dog’s name." #five

prompt+=c_none_p
prompt+="""
Question:
"""

prompt+=question_generator(ncf_num,"whistle",whistle_outcome)
prompt+="""
Answer (respond with only a number between 1 and 7 inclusive, with 1 being "not at all" and 7 being "very well"):"""


sum_int=0
for n in range(samples_per_q):
    
    raw_response=raw_response=chat_completion_with_backoff(prompt, **kwargs)
    

    word_response = raw_response['choices'][0]['message']['content']

    digit_response = int(re.search(r'\d+', word_response).group())

    sum_int+=digit_response

sum_int_avg=(sum_int)/samples_per_q #this is percent yeses for this q


total_points=[0,0,0,0,0] #history, goal, action, structure, uncompromised

total_points[4]=sum_int_avg #uncompromised


response=round(sum_int_avg,0)

prompt+=" "+str(response)+"\n"

options=[]

options.append(c_hist_p)
options.append(c_struc_p)
options.append(c_goal_p)
options.append(c_act_p)

option_nums=[]

#history, goal, action, structure, uncompromised

option_nums.append(0)
option_nums.append(3)
option_nums.append(1)
option_nums.append(2)

option_names=[]

option_names.append("c_hist_p")
option_names.append("c_struc_p")
option_names.append("c_goal_p")
option_names.append("c_act_p")



for i in range(4): 
    if (randomize):

        zipped_lists=list(zip(options,option_nums,option_names))

        random.shuffle(zipped_lists)

        options, option_nums, option_names = zip(*zipped_lists)

        options, option_nums, option_names = list(options), list(option_nums), list(option_names)
    
    prompt+=options.pop()

    prompt+="""
Question:
"""

    prompt+=question_generator(ncf_num,"whistle",whistle_outcome)
    prompt+="""
Answer (respond with only a number between 1 and 7 inclusive, with 1 being "not at all" and 7 being "very well"):"""

        
    sum_int=0
    n=0
    for n in range(samples_per_q):
        
        raw_response=chat_completion_with_backoff(prompt, **kwargs)
        

        word_response = raw_response['choices'][0]['message']['content']

        digit_response = int(re.search(r'\d+', word_response).group())

        sum_int+=digit_response


    sum_int_avg=(sum_int)/samples_per_q  #this is percent yeses for this q


    total_points[option_nums.pop()]=sum_int_avg #uncompromised

    response=round(sum_int_avg,0)

    prompt+=" "+str(response)+"\n"



print("Total points (history, goal, action, structure, uncompromised) is: "+str(total_points))

