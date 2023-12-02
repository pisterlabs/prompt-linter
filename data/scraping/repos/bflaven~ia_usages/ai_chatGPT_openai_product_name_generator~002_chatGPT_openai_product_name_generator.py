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
conda env remove -n ai_chatgpt_prompts

# update conda 
conda update -n base -c defaults conda

# to export requirements
pip freeze > requirements.txt

# to install
pip install -r requirements.txt


[filename]
002_chatGPT_openai_product_name_generator.py

[path]
cd /Users/brunoflaven/Documents/03_git/ia_usages/ai_chatGPT_openai_product_name_generator/

[run]
python 002_chatGPT_openai_product_name_generator.py

Source : https://platform.openai.com/docs/api-reference/introduction

- prompt_1
prompt="Product description: A website that provides IA insights\nSeed words: Features, Coding, IA, Web, Journalism, Media, Fun, Added Value.\nProduct names: FantasIA, ArIA, mamamIA, TrattorIA\n\nProduct description: a API that exposes several machine learning models  added values services.\nSeed words: NLP, Languages, writing, audio, video, images.",

- result_1
 "text": "\nProduct names: LinguoAPI, ScribblAI, SoundSage, VisualVantage, MediaMindAPI, TalkTechAPI, ImageIQ, LinguaLogicAPI",


- prompt_2
prompt="Product description: An information site on the climate transition. A site that allows you to determine where you will live around 2050 to face global warming, in particular to better resist temperature rises and scorching summers. Using geographical criteria, infrastructure and mobility (train, cars, etc.) ... it will be possible to determine a preferred and predefined profile for living green with a choice of: \n1. Live green in the calm of a place where I would not be solicited, not too close to infrastructure, where social connections take precedence over services\n2. Live green, by the water, near regions located by the sea, lakes or rivers\n3. Living green where my passion for the mountains blends with my life. Depending on this profile, determined on your criteria, it will then be possible to visualize your possible living spaces on a map, to zoom in and click on the areas selected on the basis of your criteria..\nProduct names: Green, Ecology\n\nProduct description: a API that exposes several machine learning models  added values services.\nSeed words: life, hiking, zero pollution,geographical, climate, transition, profile, geography, Live, green, by the water, sea, lakes, rivers, mountains\",

- result_1
 "text": ", living spaces, map, criteria, added values, services.\n\nProduct names: EcoAI, NatureMinds, GreenTech, GeoModel, ClimateFit, ProfilePlus, GeoLiving, AquaLife, MountainBlend, LiveMap, GreenSelect, EarthAI, ClimateGuide, NatureBase,",


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
    model="gpt-3.5-turbo-instruct",
    prompt="Product description: An information site on the climate transition. A site that allows you to determine where you will live around 2050 to face global warming, in particular to better resist temperature rises and scorching summers. Using geographical criteria, infrastructure and mobility (train, cars, etc.) ... it will be possible to determine a preferred and predefined profile for living green with a choice of: \n1. Live green in the calm of a place where I would not be solicited, not too close to infrastructure, where social connections take precedence over services\n2. Live green, by the water, near regions located by the sea, lakes or rivers\n3. Living green where my passion for the mountains blends with my life. Depending on this profile, determined on your criteria, it will then be possible to visualize your possible living spaces on a map, to zoom in and click on the areas selected on the basis of your criteria..\nProduct names: Green, Ecology\n\nProduct description: a API that exposes several machine learning models  added values services.\nSeed words: life, hiking, zero pollution,geographical, climate, transition, profile, geography, Live, green, by the water, sea, lakes, rivers, mountains",
    temperature=0.8,
    max_tokens=60,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
)

print('\n --- RESULT')
print(response)

