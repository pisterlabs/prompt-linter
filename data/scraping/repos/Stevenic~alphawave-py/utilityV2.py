import openai
import requests
import re
import time
import random
import traceback
import concurrent.futures
import threading as th
import json
import tracemalloc
import os
import linecache
import nltk
import selenium
import inspect
import importlib
from typing import Any, Dict, List, Tuple
import openai
import requests
import time
import traceback
import json
import os
from typing import Any, Dict, List, Tuple
import alphawave_pyexts.LLMClient as llm
from promptrix.Prompt import Prompt
from alphawave_agents.PromptCommand import PromptCommand
from promptrix.TextSection import TextSection
from promptrix.UserMessage import UserMessage
from alphawave.alphawaveTypes import PromptCompletionOptions
from alphawave_agents.PromptCommand import CommandSchema, PromptCommandOptions
from alphawave.AlphaWave import AlphaWave
from alphawave.DefaultResponseValidator import DefaultResponseValidator
from alphawave.JSONResponseValidator import JSONResponseValidator
from alphawave.TOMLResponseValidator import TOMLResponseValidator
from alphawave.MemoryFork import MemoryFork
openai.api_key = os.getenv("OPENAI_API_KEY")
google_key = os.getenv("GOOGLE_KEY")
google_cx = os.getenv("GOOGLE_CX")
GOOGLE = 'google'

def ask_LLM(model, gpt_message, max_tokens=100, temp=0.7, top_p=1.0, host = None, port = None, choice_set=None, display=None):
    completion = None
    response = ''
    #print(f'***** utility ask_LLL temperature {temp}')
    try:
      if not model.lower().startswith('gpt'):
          completion = llm.run_query(model, gpt_message, max_tokens, temp, top_p,
                                     choice_set=choice_set, host=host, port=port, display=display)
          if completion is not None:
              response = completion

      else:
        stream= openai.ChatCompletion.create(
            model=model, messages=gpt_message, max_tokens=max_tokens, temperature=temp, top_p=1, stop='STOP', stream=True)
        response = ''
        if stream is None:
          return response
        for chunk in stream:
          item = chunk['choices'][0]['delta']
          if 'content' in item.keys():
            if display is not None:
              display(chunk['choices'][0]['delta']['content'])
              response += chunk['choices'][0]['delta']['content']
    except:
        traceback.print_exc()
    return response

def findnth(haystack, needle, n):
    parts= haystack.split(needle, n+1)
    if len(parts)<=n+1:
        return -1
    return len(haystack)-len(parts[-1])-len(needle)

def extract_site(url):
    site = ''
    base= findnth(url, '/',2)
    if base > 2: site = url[:base].split('.')
    if len(site) > 1: site = site[-2]
    site = site.replace('https://','')
    site = site.replace('http://','')
    return site

def extract_domain(url):
    site = ''
    base= findnth(url, '/',2)
    if base > 2: domain = url[:base].split('.')
    if len(domain) > 1: domain = domain[-2]+'.'+domain[-1]
    domain = domain.replace('https://','')
    domain = domain.replace('http://','')
    return domain

def part_of_keyword(word, keywords):
    for keyword in keywords:
        if word in keyword:
            return True
    return False


def run_wave (client, input, prompt, prompt_options, memory, functions, tokenizer, max_repair_attempts=1, logRepairs=False, validator=DefaultResponseValidator()):
    # Create a wave for the prompt
    fork = MemoryFork(memory)
    for key, value in input.items():
      fork.set(key, value)

    wave = AlphaWave(client=client,
                     prompt=prompt,
                     prompt_options=prompt_options,
                     memory=fork,
                     functions=functions,
                     max_repair_attempts = max_repair_attempts,
                     tokenizer=tokenizer,
                     validator = validator,
                     logRepairs = logRepairs)

    response = wave.completePrompt()
    # Ensure response succeeded
    return {
      'type': "TaskResponse",
      'status': response['status'],
      'message': response['message']
    }
      

def get_search_phrase_and_keywords(client, query_string, model, memory, functions, tokenizer):

    prompt = Prompt([UserMessage('Text:\n\n{{$input}}\n\n. Analyze the above Text. Respond using this JSON template:\n\n{"Phrase": <rewrite of Text as an effective google search phrase>, "Keywords": [keywords in Text],"NamedEntities": [NamedEntities in text]}')])
    response_text=''
    completion = None
    schema = {
      'schema_type':'object',
      'title':'rephrase',
      'description':'rephrase search query',
      'properties':{
        'Phrase': {
          'type': 'string',
          'description': 'search phrase'
        },
        'Keywords': {
          'type': 'array',
          'description': 'list of keywords found'
        },
        'Named-Entities': {
          'type': 'array',
          'description': 'list of named-entities found'
        }
      },
      'required':['Phrase', 'Keywords', 'NamedEntities'],
      'returns':"query answer"
    }

    phrase = ''; keywords = []
    try:
        options = PromptCompletionOptions(completion_type='chat', model=model)
        response = run_wave(client, {'input':query_string}, prompt, options, memory, functions, tokenizer, validator=JSONResponseValidator(schema))
        
        #print(response)
        if type(response) == dict and 'status' in response and response['status'] == 'success':
            content = response['message']['content']
            if type(content) == dict:
                if 'Phrase' in content:
                    phrase = content['Phrase']
                if 'Keywords' in content:
                    keywords = content['Keywords']
                return phrase, keywords
    except Exception as e:
        traceback.print_exc()
    return phrase, keywords


def get_toml_search_phrase_and_keywords(client, query_string, model, memory, functions, tokenizer):

    prompt = Prompt([UserMessage(
"""Your task is to generate Keywords and a concise Phrase on the topic of the following Text.
Respond ONLY with Phrase and Keywords. Use this TOML schema for your response:
[RESPONSE]
Phrase="<concise topic phrase>"
Keywords="<keywords>"
[STOP]

Text:
{{$input}}
""")])

    response_text=''
    completion = None
    schema = {
        "Phrase": {
            "type":"string",
            "required": True,
            "meta": "<concise search phrase for question>"
        },
        "Keywords": {
            "type":"string",
            "required": True,
            "meta": "<keywords in question>"
        }
    }

    phrase = ''; keywords = []
    try:
        memory.set('input', query_string)
        options = PromptCompletionOptions(completion_type='chat', model=model)
        response = run_wave(client, {"input":query_string}, prompt, options, memory, functions, tokenizer,
                                  max_repair_attempts=2, validator=TOMLResponseValidator(schema))
        
        #print(response)
        if type(response) == dict and 'status' in response and response['status'] == 'success':
            content = response['message']['content']
            if type(content) == dict:
                if 'Phrase' in content:
                    phrase = content['Phrase']
                if 'Keywords' in content:
                    keywords = content['Keywords']
                return phrase, keywords
    except Exception as e:
        traceback.print_exc()
    return phrase, keywords


def reform(elements):
  #reformulates text extracted from a webpage by unstructured.partition_html into larger keyword-rankable chunks
  texts = [] # a list of text_strings, each of at most *max* chars, separated on '\n' when splitting an element is needed
  paragraphs = []
  total_elem_len = 0
  for element in elements:
    text = str(element)
    total_elem_len += len(text)
    if len(text) < 4: continue
    elif len(text)< 500:
      texts.append(text)
    else:
      subtexts = text.split('\n')
      for subtext in subtexts:
        if len(subtext) < 500:
          texts.append(subtext)
        else:
          texts.extend(nltk.sent_tokenize(subtext))
      
  # now reassemble shorter texts into chunks
  paragraph = ''
  total_pp_len = 0
  for text in texts:
    if len(text) + len(paragraph) < 500 :
      paragraph += ' '+text
    else:
      if len(paragraph) > 0: # start a new paragraph
        paragraphs.append(paragraph)
        paragraph=''
      paragraph += text
  if len(paragraph) > 0:
    paragraphs.append(paragraph+'.\n')
  total_pp_len = 0
  for paragraph in paragraphs:
    total_pp_len += len(paragraph)
  return paragraphs
