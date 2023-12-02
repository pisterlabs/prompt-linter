#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
[env]
# Conda Environment
# NO CONDA ENV

conda create --name ai_chatgpt_prompts python=3.9.13
conda info --envs
source activate ai_chatgpt_prompts
conda deactivate
# if needed to remove
conda env remove -n [NAME_OF_THE_CONDA_ENVIRONMENT]


# update conda 
conda update -n base -c defaults conda

[filename]
012_project_1_python_documentation_default_movie_to_emoji_chatgpt_api.py

[path]
cd /Users/brunoflaven/Documents/03_git/BlogArticlesExamples/ai_chatgpt_prompts/project_1_python_documentation_chatgpt_api/

[run]
python 012_project_1_python_documentation_default_movie_to_emoji_chatgpt_api.py

pip install Random

+ GREAT EXAMPLE FROM OPENAI.COM
https://platform.openai.com/examples






"""

import config_values.values_conf as conf
import random
import os
import openai

import json


"""

response = openai.Completion.create(
    model="text-davinci-003",
    prompt="Convert movie titles into emoji.\n\nBack to the Future: 👨👴🚗🕒 \nBatman: 🤵🦇 \nTransformers: 🚗🤖 \nStar Wars:",
    temperature=0.8,
    max_tokens=60,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=["\n"]
)
print('\n --- RESULT')
print(response)
"""
 

# treble clef
# response = u"\U0001D11E"

# star wars
# response = json.loads(
#     r'"\ud83c\udf1f\u2b50\ufe0f\ud83d\ude80\ud83c\udf0c"')
# print(response)

# personal configuration

OPENAI_ORGANIZATION = conf.OPENAI_ORGANIZATION
OPENAI_API_KEY = conf.OPENAI_API_KEY

# quick and dirty
openai.organization = OPENAI_ORGANIZATION
# PAID ONE DO NOT DISPLAY
openai.api_key = OPENAI_API_KEY


response = openai.Completion.create(
    model="text-davinci-003",
    # prompt="Convert movie titles into emoji.\n\nBack to the Future: 👨👴🚗🕒 \nBatman: 🤵🦇 \nTransformers: 🚗🤖 \nStar Wars: \Tootsie:",
    # prompt="Convert movie titles into emoji.\n\nBack to the Future: 👨👴🚗🕒 \nBatman: 🤵🦇 \nTransformers: 🚗🤖 \n\Le bon, la brute et le truand:",

    prompt="Convert movie titles into emoji. \nThe godfather: ",
    
    temperature=0.8,
    max_tokens=60,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=["\n"]
)
print('\n --- RESULT')
# print(response)
# print(response.choices)
desired_text = response.choices[0].text
print(desired_text)


