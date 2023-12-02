# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # llm
#
# > 

# %%
# | default_exp llm

# %%
# | hide
from nbdev.showdoc import *
from random import randint

# %%
# | export
from abc import ABC
from typing import Literal
import openai

from emblem.core import env_or_raise
from emblem.data import clean
from emblem.data import Chunks


# %%
#| export
def completion(text: str, model: Literal["gpt-3.5-turbo", "gpt-4"] = "gpt-4") -> str:
    if model in ["gpt-3.5-turbo", "gpt-4"]:
        if not hasattr(openai, 'key'):
            key = env_or_raise("OPENAI_API_KEY")
            openai.key = key

        # TODO: Add error handling, rate limits etc.
        completion = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": text}])
        response = completion.choices[0].message.content
    else:
        raise ValueError("Invalid model selected.")

    return response


# %%
# completion("What was the last thing I asked?")

# %%
#| export
def generate_question_prompt(text: str, n: int = 1, template=None, **kwargs) -> str:
    if template is None:
        template = """
        You are a tenured professor preparing questions to ask about the text provided.
        Given the following text snippet, try to create a unique knowledge question that 
        tests understanding of the source material. Response only with your question and 
        nothing else.

        SOURCE:
        {text}
        """
    
    replacements = {'text': text, **kwargs}
    for key, value in replacements.items():
        template = template.replace('{' + key + '}', str(value))
    
    return clean(str.encode(template))


# %%
# path = "../data/case_brief.pdf"
# chunks = Chunks.from_doc(path)

# %%
#| export
def generate_question(text: str, template=None, model: Literal["gpt-3.5-turbo", "gpt-4"] = "gpt-4", **kwargs):
    prompt = generate_question_prompt(text, template, **kwargs)
    response = completion(prompt, model)

    return response


# %%
# test_content = chunks[randint(0, len(chunks)-1)][1]
# test_content

# %%
# generate_question(test_content)

# %%
# | hide
import nbdev

nbdev.nbdev_export()

# %%
