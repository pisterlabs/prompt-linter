import requests
import logging
import torch
import openai
import os
import multiprocessing as mp
import nltk

from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from typing import List, Tuple, Dict
from core.helpers import DialogSession
from functools import lru_cache
from tenacity import retry, stop_after_attempt,	wait_exponential, wait_fixed  # for exponential backoff
from utils.utils import hashabledict


logger = logging.getLogger(__name__)


class GenerationModel(ABC):
	# used to generate text in general. e.g. could be using API, or local model
	@abstractmethod
	def generate(self, input_text, **gen_args):
		"""
		Generate text from the model.
		"""
		raise NotImplementedError

	def chat_generate(self, messages, **gen_args):
		"""
		Generate text from the model. Used for chatbot.
		"""
		raise NotImplementedError
	
	def chat_generate_batched(self, messages_list, **gen_args):
		"""
		Generate text from the model when you have multiple message histories
		"""
		raise NotImplementedError

	def _cleaned_resp(self, data, prompt) -> "List[str]":
		# default helper function to clean extract the generated text from the returned json
		logger.debug("promopt:")
		logger.debug(prompt)
		cleaned_resps = []
		for gen_resp in data:
			logger.debug("raw response:")
			logger.debug(gen_resp['generated_text'])
			cleaned_resp = gen_resp['generated_text'].strip()
			if "\n" in cleaned_resp:
				cleaned_resp = cleaned_resp[:cleaned_resp.index("\n")]
			logger.debug(f"cleaned response: {cleaned_resp}")
			cleaned_resps.append(cleaned_resp)
		return cleaned_resps
	
	def _cleaned_chat_resp(self, data, assistant_role="Persuader:", user_role="Persuadee:") -> "List[str]":
		# remove the user_role and keep the assistant_role
		# default helper function to clean extract the generated text from the returned json
		cleaned_resps = []
		for gen_resp in data:
			logger.debug("raw response:")
			logger.debug(gen_resp['generated_text'])
			cleaned_resp = gen_resp['generated_text'].strip()
			if "\n" in cleaned_resp:
				cleaned_resp = cleaned_resp[:cleaned_resp.index("\n")]
			if assistant_role in cleaned_resp:
				cleaned_resp = cleaned_resp[cleaned_resp.index(assistant_role) + len(assistant_role):].strip()
			if user_role in cleaned_resp:
				cleaned_resp = cleaned_resp[:cleaned_resp.index(user_role)].strip()
			logger.debug(f"cleaned response: {cleaned_resp}")
			cleaned_resps.append(cleaned_resp)
		return cleaned_resps


class DialogModel(ABC):
	# used to play DialogGame
	def __init__(self):
		self.dialog_acts = []
		return
	
	@abstractmethod
	def get_utterance(self, state:DialogSession, action) -> str:
		raise NotImplementedError
	
	def get_utterance_batched(self, state:DialogSession, action:int, batch:int) -> List[str]:
		raise NotImplementedError

	@abstractmethod
	def get_utterance_w_da(self, state:DialogSession, action) -> Tuple[str, str]:
		# this is used for user agent. should not be used for system agent
		raise NotImplementedError
	
	def get_utterance_w_da_from_batched_states(self, states:List[DialogSession], action=None):
		# this is used for user agent. should not be used for system agent
		raise NotImplementedError
		


class APIModel(GenerationModel):
	API_TOKEN = os.environ.get("HF_API_KEY")

	def __init__(self):
		# self.API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-j-6B"
		self.API_URL = "https://api-inference.huggingface.co/models/gpt2-large"
		self.headers: dict[str, str] = {"Authorization": f"Bearer {APIModel.API_TOKEN}"}
		self.inference_args = {
			"max_new_tokens": 100,
			"temperature": 0.7,
			"repetition_penalty": 1.2,
			"return_full_text": False
		}
		return

	def generate(self, input_text, **_args):
		data = {
			"inputs": input_text,
			"parameters": _args or self.inference_args
		}
		response = requests.post(self.API_URL, headers=self.headers, json=data)
		return response.json()


class OpenAIModel(GenerationModel):
	API_TOKEN = os.environ.get("OPENAI_API_KEY")

	def __init__(self, model_name="text-curie-001"):
		# check if model exists
		openai.api_key = OpenAIModel.API_TOKEN
		models = openai.Engine.list()
		if model_name not in [model.id for model in models.data]:
			raise ValueError(f"model {model_name} not found")
		
		self.inference_args = {
			"model": model_name,
			"max_tokens": 64,
			"temperature": 0.7,
			"echo": False,
			"n": 1,
			"stop": "\n"
		}
		return

	def _update_args(self, new_args):
		args = {**self.inference_args}
		from_cache = False
		if "max_new_tokens" in new_args:
			new_args["max_tokens"] = new_args.pop("max_new_tokens")
		if "return_full_text" in new_args:
			new_args["echo"] = new_args.pop("return_full_text")
		if "do_sample" in new_args:
			from_cache = not new_args.pop("do_sample")  # rely on caching
		if "num_return_sequences" in new_args:
			new_args["n"] = new_args.pop("num_return_sequences")
		if "repetition_penalty" in new_args:
			new_args["frequency_penalty"] = new_args.pop("repetition_penalty")
		return from_cache, {**args, **new_args}

	@lru_cache(maxsize=None)
	def _cached_generate(**parameters):
		response = openai.Completion.create(**parameters)
		return response
	
	# tried custom implementation of waiting before request, but I think openai is lying about how it calculates the rate limit
	# takes 3 trials to reach 2^3=8. Then 7 * 8 = 56 sec max. Just to safe we wait a bit more than 10 times
	@retry(wait=wait_exponential(multiplier=2, min=2, max=8), stop=stop_after_attempt(15))
	def generate(self, input_text, **_args):
		from_cache, parameters = self._update_args(_args)
		parameters["prompt"] = input_text
		if from_cache:
			response = OpenAIModel._cached_generate(**parameters)
		else:
			response = openai.Completion.create(**parameters)
		
		# format to a common format
		gen_output = []
		for resp in response.choices:
			text = resp.text
			gen_output.append({"generated_text": text})
		return gen_output
	

class OpenAIChatModel(OpenAIModel):
	def __init__(self, model_name="gpt-3.5-turbo", gen_sentences=-1):
		# check if model exists
		openai.api_key = self.API_TOKEN
		
		self.inference_args = {
			"model": model_name,
			"max_tokens": 64,
			"temperature": 0.7,
			"n": 1,
			# "stop": "\n"  # no longer need since we are using chat
			# "echo": False,
		}
		self.gen_sentences = None if gen_sentences < 0 else gen_sentences
		return
	
	def _update_args(self, new_args):
		if "stop" in new_args:
			new_args.pop("stop")
		if "echo" in new_args:
			new_args.pop("echo")
		if "return_full_text" in new_args:
			new_args.pop("return_full_text")
		return super()._update_args(new_args)
	
	def generate(self, input_text, **_args):
		logging.info("It is recommended to use chat_generate instead of generate for OpenAIChatModel")
		messages = [{
			"role": "user",
			"content": input_text
		}]
		return self.chat_generate(messages, **_args)
	
	@lru_cache(maxsize=None)
	def _cached_generate(**parameters):
		parameters["messages"] = list(parameters["messages"])
		response = openai.ChatCompletion.create(**parameters)
		return response
	
	@retry(wait=wait_exponential(multiplier=2, min=2, max=8), stop=stop_after_attempt(15))
	def chat_generate(self, messages: List[Dict], **gen_args):
		# generate in a chat format
		from_cache, parameters = self._update_args(gen_args)
		hashable_messages = [hashabledict(m) for m in messages]
		parameters["messages"] = hashable_messages
		if from_cache:
			parameters["messages"] = tuple(hashable_messages)  # list cannot be hashed, so cannot do **parameters
			response = OpenAIChatModel._cached_generate(**parameters)
		else:
			response = openai.ChatCompletion.create(**parameters)
		
		# format to a common format
		gen_output = []
		for resp in response.choices:
			text = resp['message']['content']
			if self.gen_sentences is not None:
				sentences = nltk.sent_tokenize(text)
				if len(sentences) > self.gen_sentences:
					text = " ".join(sentences[:self.gen_sentences])
			gen_output.append({"generated_text": text})
		return gen_output
	
	def chat_generate_batched(self, messages_list: List[List[Dict]], **gen_args):
		pool = mp.Pool(processes=len(messages_list))
		results = []
		for messages in messages_list:
			results.append(pool.apply_async(self.chat_generate, args=(messages,), kwds=gen_args))
		pool.close()
		pool.join()
		return [r.get() for r in results]


class AzureOpenAIModel(OpenAIModel):
	API_TOKEN = os.environ.get("MS_OPENAI_API_KEY")
	API_BASE = os.environ.get("MS_OPENAI_API_BASE")
	API_TYPE = "azure"
	API_VERSION = "2022-12-01"

	def __init__(self, model_name="chatgpt-turbo"):
		# check if model exists
		openai.api_key = AzureOpenAIModel.API_TOKEN
		openai.api_base = AzureOpenAIModel.API_BASE
		openai.api_type = AzureOpenAIModel.API_TYPE
		openai.api_version = AzureOpenAIModel.API_VERSION
		
		self.inference_args = {
			"engine": model_name,
			"max_tokens": 64,
			"temperature": 0.7,
			"echo": False,
			"n": 1,
			"stop": "\n"
		}
		return


class AzureOpenAIChatModel(AzureOpenAIModel):
	def __init__(self, model_name="chatgpt", gen_sentences=-1):
		# check if model exists
		openai.api_key = self.API_TOKEN
		openai.api_base = self.API_BASE
		openai.api_type = self.API_TYPE
		openai.api_version = "2023-03-15-preview"
		
		self.inference_args = {
			"engine": model_name,
			"max_tokens": 64,
			"temperature": 0.7,
			"n": 1,
			# "stop": "\n"  # no longer need since we are using chat
			# "echo": False,
		}
		self.gen_sentences = None if gen_sentences < 0 else gen_sentences
		return
	
	def _update_args(self, new_args):
		if "stop" in new_args:
			new_args.pop("stop")
		if "echo" in new_args:
			new_args.pop("echo")
		if "return_full_text" in new_args:
			new_args.pop("return_full_text")
		return super()._update_args(new_args)
	
	@lru_cache(maxsize=None)
	def _cached_generate(**parameters):
		parameters["messages"] = list(parameters["messages"])
		response = openai.ChatCompletion.create(**parameters)
		return response
	
	@retry(wait=wait_exponential(multiplier=2, min=2, max=8), stop=stop_after_attempt(15))
	def chat_generate(self, messages: List[Dict], **gen_args):
		# generate in a chat format
		from_cache, parameters = self._update_args(gen_args)
		hashable_messages = [hashabledict(m) for m in messages]
		parameters["messages"] = hashable_messages
		if from_cache:
			parameters["messages"] = tuple(hashable_messages)  # list cannot be hashed, so cannot do **parameters
			response = AzureOpenAIChatModel._cached_generate(**parameters)
		else:
			response = openai.ChatCompletion.create(**parameters)
		
		# format to a common format
		gen_output = []
		for resp in response.choices:
			text = resp['message']['content']
			if self.gen_sentences is not None:
				sentences = nltk.sent_tokenize(text)
				if len(sentences) > self.gen_sentences:
					text = " ".join(sentences[:self.gen_sentences])
			gen_output.append({"generated_text": text})
		return gen_output
	
	def chat_generate_batched(self, messages_list: List[List[Dict]], **gen_args):
		pool = mp.Pool(processes=len(messages_list))
		results = []
		for messages in messages_list:
			results.append(pool.apply_async(self.chat_generate, args=(messages,), kwds=gen_args))
		pool.close()
		pool.join()
		return [r.get() for r in results]
	
	def generate(self, input_text, **_args):
		messages = [{
			"role": "user",
			"content": input_text
		}]
		return self.chat_generate(messages, **_args)


class LocalModel(GenerationModel):
	def __init__(self, model_name="EleutherAI/gpt-neo-2.7B", input_max_len=512, stop_symbol="\n", cuda=True):
		self.tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side="left")
		self.model = AutoModelForCausalLM.from_pretrained(model_name)
		stop_token_ids = self.tokenizer.encode(stop_symbol)[0]
		set_seed(42)
		if cuda and torch.cuda.is_available():
			self.cuda = True
			self.model = self.model.cuda()
		else:
			self.cuda = False
		
		self.input_max_len = input_max_len
		self.inference_args = {
			"max_new_tokens": 128,
			"temperature": 0.7,
			"repetition_penalty": 1.0,
			"eos_token_id": stop_token_ids,
			"pad_token_id": self.tokenizer.eos_token_id
			# "return_full_text": False  # not available for manual generation
		}

	def generate(self, input_text:str, **gen_args):
		# override if gen_args specified
		gen_params = {**self.inference_args, **gen_args}
		inputs = self.tokenizer([input_text], return_tensors='pt', truncation=True, max_length=self.input_max_len)
		if self.cuda:
			inputs = {k: v.cuda() for k, v in inputs.items()}
		
		outputs = self.model.generate(**inputs, **gen_params)
		gen_only_outputs = outputs[:, len(inputs['input_ids'][0]):]
		gen_resps = self.tokenizer.batch_decode(gen_only_outputs, skip_special_tokens=True)

		# format output
		gen_output = []
		for resp in gen_resps:
			gen_output.append({"generated_text": resp})
		return gen_output