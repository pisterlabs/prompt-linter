# -*- coding: utf-8 -*-
"""
Created on Thu May 18 12:22:23 2023

@author: marca
"""

import openai
from openai.error import RateLimitError, InvalidRequestError, APIError
import time
import configparser
import tiktoken
import trafilatura

class SimpleBot:
    
    def __init__(self, primer, model="gpt-3.5-turbo"):
        self.openai_api_key = self._get_api_keys("config.ini")
        openai.api_key = self.openai_api_key
        self.model = model
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        if isinstance(primer, list):
            self.primer = [
                {"role": "system", "content": "You are a helpful assistant."}
            ]
            for message in primer:
                self.primer.append({"role": "user", "content": message})
        else:
            self.primer = [
                {"role": "system", "content": primer},
            ]

    def _get_api_keys(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        openai_api_key = config.get("API_KEYS", "OpenAI_API_KEY")
        return openai_api_key

    def _count_tokens(self, text):
        tokens = len(self.encoding.encode(text))
        return tokens

    def _generate_response(
        self,
        messages,
        function_desc=None,
        temperature=0.5,
        n=1,
        max_tokens=4000,
        frequency_penalty=0,
    ):
        token_ceiling = 4096
        if self.model == "gpt-4":
            max_tokens = 8000
            token_ceiling = 8000
        if self.model == "gpt-3.5-turbo-16k":
            max_tokens = 16000
            token_ceiling = 16000
    
        tokens_used = sum([self._count_tokens(msg["content"]) for msg in messages])
        tokens_available = token_ceiling - tokens_used
    
        max_tokens = min(max_tokens, (tokens_available - 100))
    
        if tokens_used + max_tokens > token_ceiling:
            max_tokens = token_ceiling - tokens_used - 10
    
        if max_tokens < 1:
            max_tokens = 1
    
        max_retries = 10
        retries = 0
        backoff_factor = 1  # Initial sleep time factor
    
        while retries < max_retries:
            
            try:
                completion_params = {
                    "model": self.model,
                    "messages": messages,
                    "n": n,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "frequency_penalty": frequency_penalty,
                }
                if function_desc is not None:
                    completion_params["functions"] = function_desc
    
                completion = openai.ChatCompletion.create(**completion_params)
    
                response = completion
                return response
            except Exception as e:
                print(e)
                retries += 1
                sleep_time = backoff_factor * (2 ** retries)  # Exponential backoff
                print(f"Server overloaded, retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
    
        print("Failed to generate prompt after max retries")
        return



    def smart_agent(self):
        self.model = "gpt-4"

    def fast_agent(self):
        self.model = "gpt-3.5-turbo"
        
    def long_agent(self):
        self.model = "gpt-3.5-turbo-16k"

    def add_primer(self, primer_text):
        self.primer.append({"role": "user", "content": primer_text})

    def chat(self, input_string: str, context_chunks: list=None):
        # Create a local copy of self.primer
        messages = self.primer.copy()

        # Append new user message
        messages.append({"role": "user", "content": f"{input_string}"})
        
        if context_chunks:
            
            memories = [{"role": "user", "content": f"Context:\n{context}"} for context in context_chunks]
            messages.extend(memories)
            
        response = self._generate_response(messages, temperature=0.1)

        return response
    
    
    
class WebpageSummaryBot(SimpleBot):
    
    def __init__(self, model="gpt-3.5-turbo-16k"):
        super().__init__( primer="You are my Webpage Summary Assistant.  Your job is to take the full, main text of a webpage, and trim it down into a summary.  Maintain all important details, while attempting to keep the summary as short as possible.  You must respond with a summary, and only a summary, no explanatory text or pleasantries.", model='gpt-3.5-turbo-16k')
    

    def _chunk_webpage_text(self, text, max_token_length=10000):
        words = text.split()
        chunks = []
        current_chunk = ""
    
        for word in words:
            # Check if adding the word to the current chunk would exceed the max_token_length
            if self._count_tokens(current_chunk + " " + word) > max_token_length:
                # If so, add the current chunk to the chunks list and start a new chunk with the current word
                chunks.append(current_chunk.strip())
                current_chunk = word
            else:
                # Otherwise, add the word to the current chunk
                current_chunk += f" {word}"
    
        # Add the last chunk to the chunks list
        if current_chunk:
            chunks.append(current_chunk.strip())
    
        return chunks

    def _summarize_text(self, input_string: str):
        # Create a local copy of self.primer
        messages = self.primer.copy()
        
        # Append new user message
        messages.append({"role": "user", "content": f"Text to summarize: {input_string}"})
        
        response = self._generate_response(messages, temperature=0.1)
        
        return response.choices[0].message.content

    def summarize_url_content(self, url: str):
        
        downloaded = trafilatura.fetch_url(url)
        webpage_text = trafilatura.extract(downloaded)
        
        if self._count_tokens(webpage_text) > 10000:
            
            chunks = self._chunk_webpage_text(webpage_text)
            
            summary = "\n".join([self._summarize_text(chunk) for chunk in chunks])

            
        else:
            
            summary = self._summarize_text(webpage_text)
                
        
        return summary
