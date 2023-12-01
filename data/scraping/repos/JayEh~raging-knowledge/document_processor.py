# -*- coding: utf-8 -*-
"""
@author: j.
"""

import os
import pickle
import json
import time
from typing import List, Tuple

import tiktoken
import pandas as pd
import numpy as np
from scipy import spatial
from datetime import datetime

import openai
import logging

# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"


# the UI interacts through this class
class AppCoordinator():
    def __init__(self):
        # all things embeddings
        self.embedding_processor = EmbeddingProcessor()
        
        # all the user chats are stored in self.chat_log
        self.chat_log_file = './chat_log.json'
        
        # the users chat/answer history
        if os.path.exists(self.chat_log_file):
            with open(self.chat_log_file, 'r') as f:
                self.chat_log = json.load(f)
        else:
            self.chat_log = []
    
    def ask_question(self, user_query, progress_update_function):
        progress_update_function("Searching document library..", 1)
        
        # find the related chunks
        top_n = self.embedding_processor.strings_ranked_by_relatedness(user_query)
        top_texts = list(top_n['text'])
        
        # summarize the chunks into knowledge base article
        progress_update_function("Documents acquired. Generating your article..", 1)
        summarized_article = self.embedding_processor.summarize_related_paragraphs(user_query, top_texts)
        
        # direct answer to the user query
        progress_update_function("The article is ready. Generating your answer..", 1)
        answer = self.embedding_processor.ask(user_query, summarized_article)
        
        # question, article, answer, results_df
        return (user_query, summarized_article, answer, top_n.to_json(orient='split'))
    
    def save_chat(self, question, article, answer, results_json):
        # we're not saving the embeddings in the chat history, so get rid of those
        results_df = pd.read_json(results_json, orient='split')
        results_df = results_df[['source_document','detection','text_len','text', 'similarity']].copy()
        
        # Create a dictionary to store the data for this chat
        chat_data = {
            'time': datetime.now().strftime("%m-%d-%Y %H:%M:%S"),
            'question': question,
            'article': article,
            'answer': answer,
            'results_df': results_df.to_json(orient='split')
        }
        
        # Add the chat data to the chat log
        self.chat_log.append(chat_data)

        # Save the chat log to a JSON file
        with open('chat_log.json', 'w') as file:
            json.dump(self.chat_log, file)
    
    def api_key_valid(self):
        # returns whether the api_key.txt file has a valid key in it
        try:
            (success, models) = self.embedding_processor.oai.get_models()
            return success
        
        except Exception as ex:
            logging.error('An error occurred: %s', str(ex), exc_info=True)
            return False
    
    def get_settings(self ):
        pass # future

    def get_embedding_status(self):
        settings = AppSettings().get_settings()
        embeddings = settings["embeddings"]
        # have any of the files been marked as having embeddings? return true if so
        return any([v for v in embeddings.values() if v == True])


class EmbeddingProcessor():
    def __init__(self):
        self.oai = OAI_API()
        
        # the embedding dataframe is saved here
        self.embeddings_path = './embeddings.json'

        if os.path.exists(self.embeddings_path):
            # embedding dataframe of the text and its associated embeddings
            self.emb_df = pd.read_json(self.embeddings_path, orient='index')
        else:
            columns=['source_document', 'detection', 'text_len', 'text', 'embedding']
            self.emb_df = pd.DataFrame(columns=columns)

        # this folder contains the documents we want to embed and use for question answering
        self.documents_path = './documents'
        
        # this is the standard encoding for for gpt
        self.encoding = tiktoken.encoding_for_model(GPT_MODEL)
        
        
        # these setting help to control the behavior of the assistant
        self.user_query_system = """You are a helpful question answering assistant. 
Your purpose is to answer User Questions based on information given in the Article.  
Any time you cannot anwer the question with the Article provided you reply 'I don't know.' and then explain the limits of your available knowledge.
Any time you can answer the User Question with the information in the Article, do so as shortly as possible.
You are always careful to explain the limits of the knowledge available in the Article."""
        
        # summarization system combines multiple search results into a single article to be used to answer the question
        self.summarization_system = """You are a helpful summarization and article writing assistant. 
Your purpose is to combine and summarize paragraphs snipped from various sources into a single article.
Keep in mind the paragraphs provided can be in any order and from any source, but must be combined into an informative article.
Combine the Source Data carefully and always be factual and unbiased, while maintaining all the relevant details. 
The purpose of the information in the Source Data will be to answer the User Question, which you will also be told. 
Do not answer the User Question directly, but do create a summarized yet detailed article that can be used to answer the User Question.
You are always careful to explain the limits of the knowledge available in the Source Data as it relates to the User Question."""
        
    def num_tokens(self, text: str) -> int:
        """Return the number of tokens in a string."""
        return len(self.encoding.encode(text))
    
    def has_embeddings(self, document_name: str):
        return document_name in self.emb_df['source_document'].values
    
    def split_text_into_chunks(self, text: str, max_tokens: int = 300) -> List[str]:
        paragraphs = text.split("\n")
        chunks = []
        current_chunk = ""
        current_token_count = 0
    
        for paragraph in paragraphs:
            paragraph_token_count = self.num_tokens(paragraph)
    
            if current_token_count + paragraph_token_count > max_tokens:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_token_count = 0
    
            current_chunk += paragraph + "\n"
            current_token_count += paragraph_token_count
    
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
    
        return chunks

    def __create_embedding(self, text: str) -> List[float]:
        return self.oai.embedding(text)


    def strings_ranked_by_relatedness(self, query: str, top_n: int = 6) -> tuple[list[str], list[float]]:
        cache_file = "./query_cache.pkl"

        # Load cache from disk if exists, otherwise create an empty cache
        if os.path.exists(cache_file):
        	with open(cache_file, "rb") as f:
        		cache = pickle.load(f)
        else:
        	cache = {}
        
        # Check if the query is in the cache, otherwise call the API and update the cache
        if query in cache:
        	query_embedding = cache[query]
        else:
        	query_embedding = self.oai.embedding(query)
        	cache[query] = query_embedding
        	
        	# Save the updated cache to disk
        	with open(cache_file, "wb") as f:
        		pickle.dump(cache, f)
        
        # get the np array of the embedding column
        embeddings_array = self.emb_df['embedding'].values
        
        # Calculate cosine similarities using cdist 
        cosine_similarities = 1 - spatial.distance.cdist( np.array([query_embedding]) , np.array(list(embeddings_array)) , metric='cosine' ).flatten()
    
        # Sort indices based on cosine similarities
        sorted_indices = cosine_similarities.argsort()[::-1][:top_n]
        
        # get the top results and the similarilty scores
        results = self.emb_df.iloc[sorted_indices].copy()
        results['similarity'] = cosine_similarities[sorted_indices]
        
        return results
    
    def create_embeddings(self, file_name: str):
        """
        This function will load all the text files in the self.documents_path directory
        It will then chunk the documents, allowing an approximate maximum of 500 words per document chunk while never breaking up a paragraph. 
        We will update self.emb_df, adding 1 new row for every chunk
            When adding a new row, the 'source_document' value should be the name of the file used to create the text chunk
        For every row in self.emb_df, we will take the value of the text column, encode the text with tiktoken, call openai.Embedding.create with the encoded text, and save the results to self.emb_df['embedding']
        
        Returns
        -------
        None.

        """
        
        # don't re-process anything
        if file_name in self.emb_df["source_document"].unique():
            return
        
        with open(os.path.join(self.documents_path, file_name), "r", encoding="utf-8") as file:
            text = file.read()
        
        text_chunks = self.split_text_into_chunks(text)
        
        for i, text_chunk in enumerate(text_chunks):
            if len(text_chunk) == 0:
                continue
            
            text_len = self.num_tokens(text_chunk)
            embedding = self.__create_embedding(text_chunk)
            
            row = {
                'source_document': file_name,
                'detection': i,
                'text_len': text_len,
                'text': text_chunk,
                'embedding': embedding
            }
            
            self.emb_df.loc[len(self.emb_df)] = row
            print(f"Processed chunk: {text_chunk[:50]}...")

        self.emb_df.to_json(self.embeddings_path, orient='index')
        print("Embeddings saved to", self.embeddings_path)

    def summarize_related_paragraphs(self, user_query: str, paragraphs: list):
        messages = [
            {'role': 'system', 'content': self.summarization_system },
            {'role': 'user', 'content': f"""The User Question is: ```{user_query}```
             The Source Data is: ```{str(paragraphs)}```"""}
        ]
        
        response = openai.ChatCompletion.create(
            model=GPT_MODEL,
            messages=messages,
            temperature=0.7  # for factuality temperature should be zero?
        )
        
        response_message = response["choices"][0]["message"]["content"]
        return response_message

    def ask(self, user_query: str, summarized_article: str):
        messages = [
            {'role': 'system', 'content': self.user_query_system },
            {'role': 'user', 'content': f"""The User Question: ```{user_query}```
             The Article: ```{summarized_article}```"""}
        ]
        
        response = openai.ChatCompletion.create(
            model=GPT_MODEL,
            messages=messages,
            temperature=0.7  # for factuality temperature should be zero?
        )
        
        response_message = response["choices"][0]["message"]["content"]
        return response_message

class OAI_API:
    def __init__(self):
        openai.api_key = AppSettings().get_api_key()
        self.retries = 3
    
    def embedding(self, text: str):
        # Make an OpenAI API request with retries
        for i in range(self.retries):
            try:
                response = openai.Embedding.create(
            		model=EMBEDDING_MODEL,
            		input=text,
            	)
                return np.array(response["data"][0]["embedding"])
            
            except Exception as ex:
                print(f"Error: {ex}")
                
                # authentication error means a bad api key (or none provided)
                if i < self.retries - 1 and "Invalid authorization header" != str(ex):
                    logging.debug(f"Retrying in 1 second... (retry {i + 1} of {self.retries})")
                    time.sleep(1)
                else:
                    raise
    
    def chat_completion(self, messages):
        # Make an OpenAI API request with retries
        for i in range(self.retries):
            try:
                response = openai.ChatCompletion.create(
                    model=GPT_MODEL,
                    messages=messages,
                    temperature=0.7  # for factuality temperature should be zero?
                )
                return response["choices"][0]["message"]["content"]
            
            except Exception as ex:
                print(f"Error: {ex}")
                
                # authentication error means a bad api key (or none provided)
                if i < self.retries - 1 and "Invalid authorization header" != str(ex):
                    logging.debug(f"Retrying in 1 second... (retry {i + 1} of {self.retries})")
                    time.sleep(1)
                else:
                    raise
    
    def get_models(self):
        # Make an OpenAI API request with retries
        for i in range(self.retries):
            try:
                # basic connectivity check to load the available models and ensure we can use the one we want
                response = openai.Model.list()
                models = response["data"]
                success = GPT_MODEL in [m.root for m in models]
                return (success, models)
            
            except Exception as ex:
                logging.error('An error occurred: %s', str(ex), exc_info=True)
                
                # authentication error means a bad api key (or none provided)
                if i < self.retries - 1 and "Invalid authorization header" != str(ex):
                    logging.debug(f"Retrying in 1 second... (retry {i + 1} of {self.retries})")
                    time.sleep(1)
                else:
                    raise
        

class AppSettings():
    def __init__(self):
        self.settings_file = './settings.json'
        self.api_key_file = './api_key.txt'
    
    def get_settings(self):
        # setup required files if they don't exist
        if not os.path.exists(self.settings_file):
            with open(self.settings_file, 'x') as f:
                json.dump({}, f)
        
        # Load the updated settings
        with open(self.settings_file, 'r') as f:
            settings = json.load(f)
        
        # When first load we should add this
        if 'embeddings' not in settings:
            settings['embeddings'] = {}
        
        return settings
    
    def save_settings(self, settings):
        # Save the updated settings
        with open(self.settings_file, 'w') as f:
            json.dump(settings, f)

    def get_api_key(self):
        # setup required files if they don't exist
        if not os.path.exists(self.api_key_file):
            with open(self.api_key_file, 'x') as f:
                f.write("")
        
        # Load the updated settings
        with open(self.api_key_file, 'r') as f:
            lines = f.readlines()
            if len(lines) == 0:
                logging.error(f"Expected 1 line in file {self.api_key_file}, but found {len(lines)} lines.")
                raise ValueError(f"Expected 1 line in file {self.api_key_file}, but found {len(lines)} lines.")
            else:
                # multiple lines after the first are just ignored - first line expected to be API key
                return lines[0].strip()
