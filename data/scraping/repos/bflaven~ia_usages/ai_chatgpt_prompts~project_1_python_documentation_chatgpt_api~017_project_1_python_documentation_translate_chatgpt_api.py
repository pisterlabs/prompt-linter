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
017_project_1_python_documentation_translate_chatgpt_api.py

[path]
cd /Users/brunoflaven/Documents/03_git/BlogArticlesExamples/ai_chatgpt_prompts/project_1_python_documentation_chatgpt_api/


[run]
python 017_project_1_python_documentation_translate_chatgpt_api.py

pip install Random

+ GREAT EXAMPLES FROM OPENAI.COM
https://platform.openai.com/examples

"""

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


# response=openai.Completion.create(
#         model="text-davinci-003",
#         prompt="Translate this into 1. French, 2. Spanish and 3. Japanese:\n\How many children do you have?\n\n1.",
#         temperature=0.3,
#         max_tokens=100,
#         top_p=1.0,
#         frequency_penalty=0.0,
#         presence_penalty=0.0
#     )
response=openai.Completion.create(
        model="text-davinci-003",
        prompt="Translate this into: 1. French, 2. Spanish, 3. Portuguese, 4. Italian:\n\How many children do you have?\n\n1.",
        temperature=0.3,
        max_tokens=100,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

print('\n --- RESULT')
# print(response)
# print(response.choices)
desired_text = response.choices[0].text
print(desired_text)


