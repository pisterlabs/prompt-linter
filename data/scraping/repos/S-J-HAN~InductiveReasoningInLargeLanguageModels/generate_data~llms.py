from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from dataclasses import dataclass

import openai
import cohere
import os
import requests
import time
import config

import prompts

import numpy as np

@dataclass(frozen=True)
class LLMRating:
    raw_completion: str
    parsed_rating: Optional[float]


class LLMReasoner(ABC):
    """Query LLM APIs"""

    def __init__(self, model: str):
        self.model = model
        self.key = os.environ[self.vendor.upper()]

    @property
    def vendor(self) -> str:
        return ""
    
    @property
    def api_type(self) -> str:
        return ""
    
    @property
    def name(self) -> str:
        return f"{self.model}-{self.vendor}-{self.api_type}"
    
    @abstractmethod
    def generate_response(
        self,
        prompt: Any,
        temperature: int = config.DEFAULT_TEMPERATURE,
    ) -> str:
        """Pass a prompt to LLM API and return its response as a string"""
        pass

    def generate_rating(
        self,
        prompt: Any,
        is_experiment_2: bool,
        temperature: int = config.DEFAULT_TEMPERATURE,
        num_tries: int = 5,
        sleep_time: float = 10,
    ) -> Optional[LLMRating]:
        """Make multiple attempts at getting a rating from an LLM API"""
        for _ in range(num_tries):
            try:
                llm_rating = self._generate_rating(prompt, is_experiment_2, temperature)
                time.sleep(config.SLEEP_TIMES[self.vendor])
                return llm_rating
            except:
                time.sleep(sleep_time)
        return None
    
    @abstractmethod
    def _generate_rating(
        self,
        prompt: Any,
        is_experiment_2: bool,
        temperature: int,
    ) -> LLMRating:
        """Pass an prompt that asks for a rating to LLM API and return its raw response and parsed/processed rating"""
        pass

    @classmethod
    def calculate_e1_completion_rating(cls, completion: Dict) -> Optional[float]:
        """Given a completion dictionary from OpenAI Completions endpoint, generates a score between 0 and 5 by
        taking a probability-weighted average for the answer's letter token within the completion
        This always works for text-davinci-003, but doesn't really work for 002 and 001, which tend to have underspecified answers.
        """
        
        options = prompts.E1_OPTIONS

        text = completion["choices"][0]["text"]
        i = None
        for option in options:
            if option in text:
                i = text.index(option)
                break

        if i == None:
            return None

        tokens = completion["choices"][0]["logprobs"]["tokens"]
        token_position = 0
        for i,t in enumerate(tokens):
            if text.startswith(option) or text.startswith(" " + option):
                token_position = i
                break
            text = text[len(t):]

        option_logprobs = completion["choices"][0]["logprobs"]["top_logprobs"][token_position]
        option_probs = {k: np.exp(v) for k,v in option_logprobs.items()}
        s = sum(option_probs.values())
        option_probs = {k: v/s for k,v in option_probs.items()}

        output = 0
        for i,l in enumerate("ABCDE"):
            if f" {l}" in option_probs:
                output += i*option_probs[f" {l}"]
            elif l in option_probs:
                output += i*option_probs[l]

        return output
    
    @classmethod
    def parse_e1_chat_rating(cls, response: str) -> Optional[float]:
        """
        Given a discrete response to a prompt that asks for a multi choice rating, find the rating within the response and return as a float.
        """
        options = prompts.E1_OPTIONS
        
        for option in options:
            # This if logic looks a little dicey but is written based on a visual inspection of the cases where the first condition alone fails to capture all responses
            if option in response or f"({option[0]})" in response or option.split("-")[1] in response:
                return options.index(option)
        
        return None

    @classmethod
    def calculate_e2_completion_rating(cls, completion_logprobs: Dict[str, float]) -> Optional[float]:
        """Given a dictionary of completion probabilities for numerical ratings at a single token position, 
        converts this into a single rating by taking a probability-weighted average."""
        
        completion_probs = {k: np.exp(v) for k,v in completion_logprobs.items()}
        s = sum(completion_probs.values())
        completion_probs = {k: v/s for k,v in completion_probs.items()}

        rating = 0
        c = 0
        for k,v in completion_probs.items():
            try:
                rating += float(k)*v
            except:
                c += 1
        
        if c == len(completion_logprobs):
            return None
        else:
            return rating

    @classmethod
    def parse_e2_chat_rating(cls, response: str) -> Optional[float]:
        """
        Given a discrete response to a prompt that asks for a numerical rating, find the rating within the response and return as a float.
        Ideally the entire response is just the rating. If not, look for string matches to potential ratings.
        If there are multiple parsed ratings and one of them is 100, disregard the 100 (example: 'this argument's strength is 1 out of 100', the rating is '1' not '100').
        If there are still multiple parsed ratings, take the average (this never actually happens)
        """

        split_tokens = response.split(" ")
        split_tokens = [st.strip(",").strip(".") for st in split_tokens]
        floats = []
        for st in split_tokens:
            try:
                floats.append(float(st))
            except:
                pass
        if len(floats) == 1:
            return floats[0]
        floats = [f for f in floats if f != 100]
        if len(floats) == 1:
            return floats[0]
        elif len(floats) > 1:
            return np.mean(floats)
        else:
            return None


class OpenAIChatReasoner(LLMReasoner):
    """'Chat' style OpenAI models like GPT-4 and GPT-3.5-chat-turbo"""

    @property
    def vendor(self) -> str:
        return "openai"
    
    @property
    def api_type(self) -> str:
        return "chat"
    
    def generate_response(
        self,
        prompt: Any,
        temperature: int = config.DEFAULT_TEMPERATURE,
    ) -> str:
        
        openai.api_key = self.key
        completion = openai.ChatCompletion.create(
              model=self.model,
              messages=prompt,
              temperature=temperature,
            )

        return completion.choices[0].message.content

    def _generate_rating(
        self, 
        prompt: prompts.ChatMessage, 
        is_experiment_2: bool,
        temperature: int = config.DEFAULT_TEMPERATURE,
    ) -> LLMRating:

        assistant_message = self.generate_response(prompt, temperature)
        if not is_experiment_2:
            rating = self.parse_e1_chat_rating(assistant_message)
        else:
            rating = self.parse_e2_chat_rating(assistant_message)

        return LLMRating(assistant_message, rating)
    

class OpenAICompletionReasoner(LLMReasoner):
    """'Completion' style OpenAI models like text-danvinci-003, 002, 001"""

    @property
    def vendor(self) -> str:
        return "openai"
    
    @property
    def api_type(self) -> str:
        return "completion"
    
    def generate_response(
        self,
        prompt: Any,
        temperature: int = config.DEFAULT_TEMPERATURE,
    ) -> str:
        
        openai.api_key = os.environ['OPENAI']
        completion = openai.Completion.create(
            model=self.model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=100,
            logprobs=5,
        )

        return completion["choices"][0]["text"]

    def _generate_rating(
        self, 
        prompt: str,
        is_experiment_2: bool,
        temperature: int = config.DEFAULT_TEMPERATURE,
    ) -> LLMRating:

        openai.api_key = os.environ['OPENAI']
        completion = openai.Completion.create(
            model=self.model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=100,
            logprobs=5,
        )

        if not is_experiment_2:
            rating = self.calculate_e1_completion_rating(completion["choices"][0].logprobs.top_logprobs[-1])
        else:
            rating = self.calculate_e2_completion_rating(completion["choices"][0].logprobs.top_logprobs[-1])

        return LLMRating(completion, rating)


class CohereCompletionReasoner(LLMReasoner):

    @property
    def vendor(self) -> str:
        return "cohere"
    
    @property
    def api_type(self) -> str:
        return "completion"
    
    def generate_response(
        self,
        prompt: Any,
        temperature: int = config.DEFAULT_TEMPERATURE,
    ) -> str:
        co = cohere.Client(os.environ['COHERE'])
        return co.generate(
            model=self.model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=100,
        ).generations[0].text

    def _generate_rating(
        self, 
        prompt: str,
        is_experiment_2: bool,
        temperature: int = config.DEFAULT_TEMPERATURE,
    ) -> LLMRating:

        prediction = self.generate_response(prompt, temperature)
        if not is_experiment_2:
            rating = self.parse_e1_chat_rating(prediction)
        else:
            rating = self.parse_e2_chat_rating(prediction)

        return LLMRating(prediction, rating)
    

class TextSynthCompletionReasoner(LLMReasoner):

    @property
    def vendor(self) -> str:
        return "textsynth"

    @property
    def api_type(self) -> str:
        return "completion"
    
    def generate_response(
        self,
        prompt: Any,
        temperature: int = 0.11,
    ) -> str:
        api_url = "https://api.textsynth.com"
        response = requests.post(api_url + "/v1/engines/" + self.model + "/completions", headers = { "Authorization": "Bearer " + os.environ['TEXTSYNTH'] }, json = { "prompt": prompt, "max_tokens": 100, "temperature": temperature })
        resp = response.json()
        if "text" in resp: 
            return resp["text"]
        else:
            return None

    def _generate_rating(
        self, 
        prompt: str,
        is_experiment_2: bool,
        temperature: int = 0.11,
    ) -> LLMRating:
        
        completion = self.generate_response(prompt, temperature)
        if not is_experiment_2:
            rating = self.parse_e1_chat_rating(completion)
        else:
            rating = self.parse_e2_chat_rating(completion)

        return LLMRating(completion, rating)
