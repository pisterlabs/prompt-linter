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
python 013_project_1_python_documentation_parse_unstructured_data_chatgpt_api.py

pip install Random

+ GREAT EXAMPLE FROM OPENAI.COM
https://platform.openai.com/examples

"""

import random
import os
import openai
import json


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
    

    # prompt="A table summarizing the fruits from Goocrux:\n\nThere are many fruits that were found on the recently discovered planet Goocrux. There are neoskizzles that grow there, which are purple and taste like candy. There are also loheckles, which are a grayish blue fruit and are very tart, a little bit like a lemon. Pounits are a bright green color and are more savory than sweet. There are also plenty of loopnovas which are a neon pink flavor and taste like cotton candy. Finally, there are fruits called glowls, which have a very sour and bitter taste which is acidic and caustic, and a pale orange tinge to them.\n\n| Fruit | Color | Flavor |",


prompt="A table summarizing the ingredients from Tzatziki:\n\nTzatziki (Greek: τζατζίκι), also known as cacık (Turkish pronunciation: [dʒaˈdʒɯk]) or tarator, is a class of dip, soup, or sauce found in the cuisines of Southeastern Europe and the Middle East. It is made of salted strained yogurt or diluted yogurt mixed with cucumbers, garlic, salt, olive oil, sometimes with vinegar or lemon juice, and herbs such as dill, mint, parsley and thyme. It is served as a cold appetizer (mezze), a side dish, and as a sauce for souvlaki and gyros sandwiches and other foods.\n\n| Ingredients | Color | Flavor |",

    # https://en.wikipedia.org/wiki/Tzatziki

    temperature=0,
    max_tokens=100,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
)
print('\n --- RESULT')
print(response)
# print(response.choices)
# desired_text = response.choices[0].text
# # print(desired_text)


