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
010_project_1_python_documentation_default_translate_code_chatgpt_api.py

[path]
cd /Users/brunoflaven/Documents/01_work/blog_articles/ai_chatgpt_prompts/project_1_python_documentation_chatgpt_api/


[run]
python 011_project_1_python_documentation_default_notes_to_summary_chatgpt_api.py

pip install Random

+ GREAT EXAMPLE FROM OPENAI.COM
https://platform.openai.com/examples




"""

import random
import os
import openai


# personal configuration
import config_values.values_conf as conf

OPENAI_ORGANIZATION = conf.OPENAI_ORGANIZATION
OPENAI_API_KEY = conf.OPENAI_API_KEY

# quick and dirty
openai.organization = OPENAI_ORGANIZATION
# PAID ONE DO NOT DISPLAY
openai.api_key = OPENAI_API_KEY


response = openai.Completion.create(
    model="text-davinci-003",
    prompt="Convert my short hand into a first-hand account of the meeting:\n\nTom: Profits up 50%\nJane: New servers are online\nKjel: Need more time to fix software\nJane: Happy to help\nParkman: Beta testing almost done",
    temperature=0,
    max_tokens=64,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
)

print('\n --- RESULT')
print(response)
