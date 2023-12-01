import os
import re
import json
import glob
import openai
import numpy as np
from huggingface_hub import InferenceClient
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import transformers
import torch
import time

import dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import multiprocessing as mp

dotenv.load_dotenv(override=True)
openai.api_key = os.getenv('OPENAI_API_KEY')
openai.organization = os.getenv('OPENAI_ORGANIZATION')
if os.getenv('OPENAI_API_TYPE') is not None:
  openai.api_type = os.getenv('OPENAI_API_TYPE')
if os.getenv('OPENAI_API_BASE') is not None:
  openai.api_base = os.getenv('OPENAI_API_BASE')
if os.getenv('OPENAI_API_VERSION') is not None:
  openai.api_version = os.getenv('OPENAI_API_VERSION')

LLAMA_TEMPLATE = """<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_message} [/INST] """


quantization_config_4bit = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

def get_hf_model_name(model_name):
  if model_name == 'llama2-70b-chat':
    return '4bit/Llama-2-70b-chat-hf'
  elif model_name == 'llama2-13b-chat':
    return '4bit/Llama-2-13b-chat-hf'
  elif model_name == 'llama2-7b-chat':
    return '4bit/Llama-2-7b-chat-hf'
  else:
    return model_name 

def load_hf_model(model_name, quantization_config):
  model_name = get_hf_model_name(model_name)
  
  if not torch.cuda.is_available():
    quantization_config = None

  try:
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, device_map="auto")
  except:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, quantization_config=quantization_config, device_map="auto")

  tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')

  model.config.eos_token_id = tokenizer.eos_token_id
  if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=16)
    model.config.pad_token_id = model.config.eos_token_id
  
  return model, tokenizer

@retry(
    reraise=True,
    stop=stop_after_attempt(20),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=(retry_if_exception_type(openai.error.Timeout)
        | retry_if_exception_type(openai.error.APIError)
        | retry_if_exception_type(openai.error.APIConnectionError)
        | retry_if_exception_type(openai.error.RateLimitError)),

)
def chat_decode(input: list, max_length: int = 128, temp: float = 0, stop: str | list[str] | None = None, n: int = 1, engine='gpt-4'):

    if openai.api_type == 'azure':
      response = openai.ChatCompletion.create(
        engine=engine,
        messages=input,
        max_tokens=max_length,
        temperature=temp,
        stop=stop,
        n=n)
    else:
      response = openai.ChatCompletion.create(
        model=engine,
        messages=input,
        max_tokens=max_length,
        temperature=temp,
        stop=stop,
        n=n)
    
    return [response["choices"][j]["message"]["content"] for j in range(len(response["choices"]))]

def get_template(template_name):
  templates = {}
  dirname = os.path.dirname(__file__)
  for _f in glob.glob(os.path.join(dirname, 'templates', '*.json')):
    with open(_f, 'r') as f:
      _templates = {t['name']: t for t in json.load(f)}
    templates.update(_templates)

  return templates[template_name]
  
class LMRM():
  def __init__(self, model: str | type[transformers.PreTrainedModel], tokenizer: type[transformers.PreTrainedTokenizer] = None, model_type: str = 'api', template: str | dict = 'basic_template', max_score = 7, use_cot: bool = True, max_parallel: int = 5, meta_template: str = LLAMA_TEMPLATE, quantization_config: type[BitsAndBytesConfig] = quantization_config_4bit, temperature: float = 1.):
    self.model = None
    self.model_type = model_type
    self.openai = False
    self.use_cot = use_cot
    self.max_parallel = max_parallel
    self.tokenizer = None
    self.meta_template = meta_template
    self.max_score = max_score
    self.temperature = temperature

    if not type(model) == str:
      print('Using local model...')
      self.model_type = 'local' 
      self.model = model
      self.tokenizer = tokenizer
      assert tokenizer.padding_side == 'left'
    else:
      if model_type == 'openai':
        models = [m['id'] for m in openai.Model.list()['data']]
        try:
          openai.Model.retrieve(model)
          self.model = model
          self.openai = True
          print(f"Found {model} in OpenAI directory! Treating as OpenAI model.")
        except openai.error.AuthenticationError as e:
          print(e)
          raise e
        except Exception as e:
          if model in models:
            raise ValueError(f"Model {model} is an OpenAI model but is not available. Please use a HuggingFace model instead.")
          raise e
      else:
        assert model_type in ['api', 'local']
        print(f"Treating {model} as a HuggingFace model")

        if model_type == 'api':
          self.model = InferenceClient(model=model, token=os.getenv('HUGGINGFACE_TOKEN'))
        elif model_type == 'local':
          print('Loading local model...')
          self.model, self.tokenizer = load_hf_model(model, quantization_config)

        else:
          raise ValueError('Invalid model type')

    if isinstance(template, str):
      self.template = get_template(template)
    else:
      self.template = template
    self.template['logit_template'] = self.template['logit_template'].replace('{max_score}', str(self.max_score))

    if self.tokenizer is not None:
      tokenizer = self.tokenizer
      self.score_tokens = [
        tokenizer('1', add_special_tokens=False), 
        tokenizer('2', add_special_tokens=False), 
        tokenizer('3', add_special_tokens=False), 
        tokenizer('4', add_special_tokens=False), 
        tokenizer('5', add_special_tokens=False), 
        tokenizer('6', add_special_tokens=False), 
        tokenizer('7', add_special_tokens=False)
        ][:int(self.max_score)]
      self.score_tokens = [t['input_ids'][-1] for t in self.score_tokens]

  def score(self, conversations: str | list[str]):
    if self.model_type == 'local':
      if type(conversations) == str:
        conversations = [conversations]
        return self.score_local(conversations, return_scalar=True)
      return self.score_local(conversations)
    
    if isinstance(conversations, list):
      with mp.Pool(min(self.max_parallel,  len(conversations))) as pool:
        if self.openai:
          return pool.map(self.score_openai, conversations)
        else:
          return pool.map(self.score_hf_api, conversations)
  
    if self.openai:
      return self.score_openai(conversations)
    else:
      return self.score_hf_api(conversations)
  
  def score_openai(self, conversation: str):
    user_message = self.template['argmax_score_template'] if self.use_cot else self.template['argmax_score_template_no_cot']
    user_message = user_message.format(conversation = conversation)
    messages = [
      {"role": "system", "content": self.template['system_prompt']},
      {"role": "user", "content": user_message}
    ]

    response = chat_decode(messages, engine=self.model)
    res = re.search('\[\[(.*)\]\]', response[0])
    try:
      res = float(res.group(1))
      assert 0 <= res <= 10
      return res
    except Exception as e:
      print(f"GPT did not return a proper score. Exception: {e}. GPT's full response: {response[0]}")
      return None
    
  def score_hf_api(self, conversations: str | list[str]):
    user_message = self.template['logit_template'].format(conversation = conversations)
    prompt = self.meta_template.format(system_prompt=self.template['system_prompt'], user_message=user_message)
    prompt += self.template['logit_completion_template']

    for _ in range (5):
      try:
        output = self.model.post(json={'inputs': prompt, 'parameters':{'top_n_tokens': 5, 'details': True, 'max_new_tokens': 1}})
        break
      except Exception as e:
        print(f"Exception {e} occurred. Retrying...")
        time.sleep(5)

    top_tokens = json.loads(output)[0]['details']['top_tokens'][0]
    top_tokens = {t['text']:t['logprob'] for t in top_tokens}
    scores = np.array([top_tokens.get(str(i), -100.) for i in range(1,self.max_score+1)])    

    scores = scores / self.temperature
    scores -= np.max(scores)
    scores = np.exp(scores)
    scores = scores / np.sum(scores)
    return np.arange(self.max_score).dot(scores) * 9/(self.max_score - 1) + 1 # normalize to 1-10

  def score_local(self, conversations: list[str], return_scalar: bool = False):
    model = self.model
    user_message = [self.template['logit_template'].format(conversation = c) for c in conversations]
    prompt = [self.meta_template.format(system_prompt=self.template['system_prompt'], user_message=u) for u in user_message]
    prompt = [p + self.template['logit_completion_template'] for p in prompt]
    
    with torch.no_grad():
      inputs = self.tokenizer(prompt, return_tensors='pt', padding=True).to(model.device)

      logits = torch.log_softmax(model(**inputs)['logits'].to(torch.float32), -1)
      logprobs = logits[:, -1, self.score_tokens].to('cpu').numpy()
      
      scores = logprobs / self.temperature
      scores -= np.max(scores, axis=-1, keepdims=True)
      scores = np.exp(scores)
      scores = scores / np.sum(scores, axis=-1, keepdims=True)
      scores = np.arange(self.max_score).dot(scores.T) * 9/(self.max_score - 1) + 1

      if return_scalar:
        return scores[0]
      return scores

