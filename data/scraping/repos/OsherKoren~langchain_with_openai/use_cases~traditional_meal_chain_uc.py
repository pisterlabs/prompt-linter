# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""This module is for running the traditional meal chain use case."""

import warnings
warnings.filterwarnings("ignore")

from langchain.chains import LLMChain, SimpleSequentialChain

import models, prompts


def run_location_model():
    prompt_template = prompts.setup_location_template_prompt()
    llm = models.set_openai_model(max_tokens=500)
    return LLMChain(llm=llm, prompt=prompt_template)


def run_meal_model():
    prompt_template = prompts.setup_meal_template_prompt()
    llm = models.set_openai_model(max_tokens=500)
    return LLMChain(llm=llm, prompt=prompt_template)


if __name__ == "__main__":
    location_llm = run_location_model()
    meal_llm = run_meal_model()
    llm_chain = SimpleSequentialChain(chains=[location_llm, meal_llm], verbose=True)
    response = llm_chain.run("Black Forest, Germany")
    print(response)
