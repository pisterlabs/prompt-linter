openai_organization = "" #use your openai organization id
openai_api_key = "" #use your openai api key
CACHE = "./cache/" #change to your huggingface cache folder (where the models will be downloaded)
offload_folder = "./offload_folder/" #change to your offload folder (empty temporary dir for offloading huge models)
selected = ['ar', 'ca', 'cs', 'de', 'en', 'es', 'nl', 'pt', 'ru', 'uk', 'zh']
testing=False

import sys

MODEL = sys.argv[1]
DATASET = sys.argv[2]

import os
os.environ['HF_HOME'] = CACHE

import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, T5ForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import torch
import gc
import time
import shutil
import nvidia_smi, psutil
from langcodes import *
from tqdm import tqdm
import backoff
import openai
openai.organization = openai_organization
openai.api_key = openai_api_key

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError))
def completions_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)

@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError))
def chat_completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

model_name = MODEL.split("/")[-1]
if os.path.isfile(DATASET.replace('.csv', f'_{model_name}.csv')):
  shutil.copy(DATASET.replace('.csv', f'_{model_name}.csv'), DATASET.replace('.csv', f'_{model_name}.csv') + '_' + str(int(time.time())))
  df = pd.read_csv(DATASET.replace('.csv', f'_{model_name}.csv'))
else:
  df = pd.read_csv(DATASET)

if testing:
    #subsampling for testing purpose
    subset = df.groupby('language').sample(5, random_state=0).reset_index(drop=True)
    subset['selected'] = [x in selected for x in subset.language]
    subset = subset[subset.selected]
    subset = subset.drop(columns=['selected'])

    #just single sample for testing
    #subset = subset[10:11]
else:
    subset = df
    subset['selected'] = [x in selected for x in subset.language]
    subset = subset[subset.selected]
    subset = subset.drop(columns=['selected'])

use4bit = False
use8bit = False
if ("lora" not in MODEL):
    use4bit = True
    quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True, load_in_4bit=use4bit, load_in_8bit=use8bit, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
else:
    use8bit = True
    quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True, load_in_4bit=use4bit, load_in_8bit=use8bit)
    
if ("gpt-3.5-turbo" in MODEL) or ("text-davinci-003" in MODEL) or ("gpt-4" in MODEL):
  pass
elif "lora" in MODEL:
  config = PeftConfig.from_pretrained(MODEL)
  if "flan-ul2" in MODEL:
    model = T5ForConditionalGeneration.from_pretrained(config.base_model_name_or_path, device_map='auto', quantization_config=quantization_config, offload_state_dict=True, max_memory={0: "20GIB", "cpu": "50GIB"}, offload_folder=offload_folder, load_in_4bit=use4bit, load_in_8bit=use8bit, torch_dtype=torch.float16, cache_dir=CACHE)
  else:
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map='auto', quantization_config=quantization_config, offload_state_dict=True, max_memory={0: "20GIB", "cpu": "50GIB"}, offload_folder=offload_folder, load_in_4bit=use4bit, load_in_8bit=use8bit, torch_dtype=torch.float16, cache_dir=CACHE)
  if "llama" in config.base_model_name_or_path:
    tokenizer = LlamaTokenizer.from_pretrained(config.base_model_name_or_path, cache_dir=CACHE)
    # unwind broken decapoda-research config - based on https://github.com/tloen/alpaca-lora/blob/683810b4a171aa2215047311841c4a4056a5cecb/generate.py#L71
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
  else:
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, cache_dir=CACHE)
  model = PeftModel.from_pretrained(model, MODEL, offloaf_dir=offload_folder, torch_dtype=torch.float16, device_map='auto')
elif "flan-ul2" in MODEL:
  tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=CACHE)
  model = T5ForConditionalGeneration.from_pretrained(MODEL, device_map='auto', offload_state_dict=True, max_memory={0: "20GIB", "cpu": "50GIB"}, offload_folder=offload_folder, load_in_4bit=use4bit, load_in_8bit=use8bit, torch_dtype=torch.float16, cache_dir=CACHE)
else:
  tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=CACHE)
  model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, device_map='auto', offload_state_dict=True, max_memory={0: "20GIB", "cpu": "50GIB"}, offload_folder=offload_folder, load_in_4bit=use4bit, load_in_8bit=use8bit, torch_dtype=torch.float16, cache_dir=CACHE)

generated = [""] * len(subset)
if ("gpt-3.5-turbo" in MODEL) or ("text-davinci-003" in MODEL) or ("gpt-4" in MODEL):
  pass
else:
  model = model.eval()

model_name = MODEL.split("/")[-1]
if testing: model_name = model_name + "_testing2"

with torch.no_grad():
  for index, row in tqdm(subset.iterrows(), total=subset.shape[0]):
    if ("generated" in row.index) and (row['generated'] is not np.NaN) and (str(row['generated']) != "nan"):
      generated[index] = row['generated']
      #print(index, 'skipping')
      continue
    #for testing purpose
    else:
      #print(index, 'processing')
      #continue
      pass
    if ("opt" in MODEL) or ("bloom-" in MODEL):
      #generation based on title using advanced sampling strategy
      prompt = row.title
      input_ids = tokenizer(prompt, return_tensors="pt").input_ids
      with torch.cuda.amp.autocast():
        generated_ids = model.generate(input_ids=input_ids.to(device), min_length = 200, max_length = 512, num_return_sequences=1, do_sample=True, num_beams=1, top_k=50, top_p=0.95)
      result = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
      generated[index] = result[0]
      subset['generated'] = generated
      subset.to_csv(DATASET.replace('.csv', f'_{model_name}.csv'), index=False)
    else:
      language = row.language
      language_name = Language.make(language=row.language).display_name()
      headline = row.title
      prompt = f'You are a multilingual journalist.\n\nTask: Write a news article in {language_name} for the following headline: "{headline}". Leave out the instructions, return just the text of the article.\n\nOutput:'
      if ("gpt-3.5-turbo" in MODEL) or ("gpt-4" in MODEL):
        result = chat_completions_with_backoff(model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=512, top_p=0.95).choices[0].message.content
      elif ("text-davinci-003" in MODEL):
        result = completions_with_backoff(model=MODEL, prompt=prompt, max_tokens=512, top_p=0.95).choices[0].text
      else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        with torch.cuda.amp.autocast():
          generated_ids = model.generate(input_ids=input_ids.to(0), min_length = 200, max_length = 512, num_return_sequences=1, do_sample=True, num_beams=1, top_k=50, top_p=0.95)
        result = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        result = result[0]
      generated[index] = result
      subset['generated'] = generated
      subset.to_csv(DATASET.replace('.csv', f'_{model_name}.csv'), index=False)
