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
003_project_1_python_documentation_default_product_name_gen_chatgpt_api.py

[path]
cd /Users/brunoflaven/Documents/03_git/BlogArticlesExamples/ai_chatgpt_prompts/project_1_python_documentation_chatgpt_api/

[run]
python 003_project_1_python_documentation_default_product_name_gen_chatgpt_api.py

Source : https://platform.openai.com/docs/api-reference/introduction



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




response = openai.Completion.create(
    model="text-davinci-003",
    prompt="Product description: A home milkshake maker\nSeed words: fast, healthy, compact.\nProduct names: HomeShaker, Fit Shaker, QuickShake, Shake Maker\n\nProduct description: A pair of shoes that can fit any foot size.\nSeed words: adaptable, fit, omni-fit.",
    temperature=0.8,
    max_tokens=60,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
)

print('\n --- RESULT')
print(response)
