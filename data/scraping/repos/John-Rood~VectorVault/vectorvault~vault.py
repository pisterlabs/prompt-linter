# VECTOR VAULT CONFIDENTIAL
# __________________
# 
#  All Rights Reserved.
# 
# NOTICE:  All information contained herein is, and remains
# the property of Vector Vault and its suppliers,
# if any.  The intellectual and technical concepts contained
# herein are proprietary to Vector Vault
# and its suppliers and may be covered by U.S. and Foreign Patents,
# patents in process, and are protected by trade secret or copyright law.
# Dissemination of this information or reproduction of this material
# is strictly forbidden unless prior written permission is obtained
# from Vector Vault. See license for consent.

import numpy as np
import tempfile
import os
import time
import uuid
import re
import json
import traceback
import random
from typing import Union, List
from concurrent.futures import ThreadPoolExecutor
from .cloudmanager import CloudManager
from .ai import AI, openai
from .itemize import itemize, name_vecs, get_item, get_vectors, build_return, cloud_name, name_map
from .vecreq import call_get_similar
from .tools_gpt import ToolsGPT


class Vault:
    def __init__(self, user: str = None, api_key: str = None, vault: str = None, openai_key: str = None, dims: int = 1536, verbose: bool = False, personality_message: str = 'Answer the Question directly and be helpful'):
        ''' 
        ### Create a vector database instance like this:
        ```
        vault = Vault(user='YOUR_EMAIL',
              api_key='VECTOR_VAULT_API_KEY',
              openai_key='OPENAI_API_KEY',
              vault='VAULT_NAME',
              verbose=True)
        ```

        ### Add data to the vector database aka "the Vault":
        ```
        vault.add('some text')
        vault.get_vectors()
        vault.save()
        ```

        ### Basic ChatGPT response:
        `basic_chatgpt_answer = vault.get_chat('some question')`

        ### Advanced RAG response: (Retrieval Augemented Generation) -> (uses 'get_context=True' to reference the Vault - [which will internally pulls 4 vector similar search results from the database to be used as context before responding])
        `rag_answer = vault.get_chat('some question', get_context=True)`

        ### Change the model to GPT4 with the `model` param in get_chat:
        `gpt4_rag_answer = vault.get_chat('some question', get_context=True, model='gpt-4')`

        ### Change the personality of your bot with the personality_message:
        ```
        vault = Vault(user='YOUR_EMAIL',
              api_key='VECTOR_VAULT_API_KEY',
              openai_key='OPENAI_API_KEY',
              vault='VAULT_NAME',
              personality_message='Answer like you are Snoop Dogg',
              verbose=False)
        ```
        '''
        if openai_key:
            self.openai_key = openai_key
            openai.api_key = self.openai_key
        self.vault = vault.strip() if vault else 'home'
        self.vectors = get_vectors(dims)
        self.api = api_key
        self.dims = dims
        self.verbose = verbose
        try:
            self.cloud_manager = CloudManager(user, api_key, self.vault)
            if self.verbose:
                print(f'Connected vault: {self.vault}')
        except Exception as e:
            print('API KEY NOT FOUND! Using Vault without cloud access. `get_chat()` will still work', e)
            # user can still use the get_chat() function without an api key
            self.cloud_manager = None
        self.user = user
        self.x = 0
        self.x_checked = False
        self.vecs_loaded = False
        self.map = {}
        self.items = []
        self.last_time = None
        self.saved_already = False
        self.ai = AI(personality_message)
        self.tools = ToolsGPT(verbose=verbose)
        self.rate_limiter = RateLimiter(max_attempts=30)

    def get_vaults(self, vault: str = None):
        '''
            Returns a list of vaults within the current vault directory 
        '''
        vault = self.vault if vault is None else vault
        return self.cloud_manager.list_vaults(vault)

    def get_total_items(self):
        '''
            Returns the total number of vectored items in the Vault
        '''
        self.check_index()
        return self.vectors.get_n_items()
    
    def get_distance(self, id1, id2):
        '''
            Returns the distance between two vectors - item ids are needed to compare
        '''
        self.check_index()
        return self.vectors.get_distance(id1, id2)
    
    def get_item_vector(self, item_id : int):
        '''
            Returns the vector from an item id
        '''
        self.check_index()
        return self.vectors.get_item_vector(item_id)
    
    def save(self, trees=16):
        '''
            Saves all the data added locally to the Cloud. All Vault references are Cloud references.
            To add data to your Vault and access it later, you must first call add(), then get_vectors(), and finally save().
        '''
        if self.saved_already:
            self.clear_cache()
            print("The last save was aborted before the build process finished. Clearing cache to start again...")
        self.saved_already = True # Make sure the if the save process is interrupted, data will not get corrupted
        start_time = time.time()
        self.vectors.build(trees)

        total_saved_items = 0

        with ThreadPoolExecutor() as executor:
            for item in self.items:
                item_text, item_id, item_meta = get_item(item)
                executor.submit(self.cloud_manager.upload, self.map.get(str(item_id)), item_text, item_meta)
                total_saved_items += 1

        self.upload_vectors()

        if self.verbose:
            print(f"upload time --- {(time.time() - start_time)} seconds --- {total_saved_items} items saved")
            
    def clear_cache(self):
        '''
            Clears the cache for all the loaded items 
        '''
        self.reload_vectors()
        self.x_checked = True
        self.vecs_loaded = True
        self.saved_already = False

    def delete(self):
        '''
            Deletes the entire Vault and all contents
        '''
        if self.verbose:
            print('Deleting started. Note: this can take a while for large datasets')
        # Clear the local vector data
        self.vectors = get_vectors(self.dims)
        self.items.clear()
        self.cloud_manager.delete()
        self.x = 0
        print('Vault deleted')
    
    def remap(self, item_id):
        for i in range(item_id, len(self.map) - 1):
            self.map[str(i)] = self.map[str(i + 1)]
        self.map.popitem()
    
    def delete_items(self, item_ids : Union[int, List[int]], trees = 16) -> None:
        '''
            Deletes one or more items from item_id(s) passed in.
            item_ids is an int or a list of integers
        '''
        def rebuild_vectors(item_id):
            '''Deletes target vector and rebuilds the database'''
            num_existing_items = self.vectors.get_n_items()
            new_index = get_vectors(self.dims)
            plus_one = 0

            for i in range(num_existing_items):
                if i == item_id:
                    # plus_one equals 0 until the item being deleting has been removed
                    plus_one += 1 
                else: 
                    vector = self.vectors.get_item_vector(i - plus_one) 
                    new_index.add_item(i - plus_one, vector)

            self.vectors = new_index
            self.vectors.build(trees)
            self.upload_vectors()

        item_ids = [item_ids] if isinstance(item_ids, int) else item_ids
        for item_id in item_ids:
            self.load_vectors()
            self.cloud_manager.delete_item(self.map[str(item_id)])
            self.remap(item_id)
            rebuild_vectors(item_id)

        if self.verbose:
            print(f'Item {item_id} deleted')

    def edit_item(self, item_id : int, new_text : str, metadata : dict = None, trees = 16) -> None:
        '''
            Edits any item. Enter the new text and new vectors will automatically be created.
            New data and vectors will be uploaded and the old data will be deleted
        '''
        def edit_vector(item_id, new_vector):
            '''Replaces old vector with new content vector'''
            num_existing_items = self.vectors.get_n_items()
            new_index = get_vectors(self.dims)

            for i in range(num_existing_items):
                if i == item_id:
                    new_index.add_item(i, new_vector)
                else: 
                    vector = self.vectors.get_item_vector(i) 
                    new_index.add_item(i, vector)

            self.vectors = new_index
            self.vectors.build(trees)
            self.upload_vectors()

        self.load_vectors()
        self.cloud_manager.upload_to_cloud(cloud_name(self.vault, self.map[str(item_id)], self.user, self.api, item=True), new_text)
        edit_vector(item_id, self.process_batch([new_text], never_stop=False, loop_timeout=180)[0])

        if metadata:
            self.cloud_manager.upload_to_cloud(self.cloud_name(self.vault, self.map[str(item_id)], self.user, self.api, meta=True), json.dumps(metadata))

        if self.verbose:
            print(f'Item {item_id} edited')

    def check_index(self):
        if not self.x_checked:
            start_time = time.time()
            if self.cloud_manager.vault_exists(name_vecs(self.vault, self.user, self.api)):
                if not self.vecs_loaded:
                    self.load_vectors()
                self.reload_vectors()
            
            self.x_checked = True
            if self.verbose:
                print("initialize index --- %s seconds ---" % (time.time() - start_time))

    def load_mapping(self):
        '''Internal only!'''
        try: # try to get the map
            temp_file_path = self.cloud_manager.download_to_temp_file(name_map(self.vault, self.user, self.api))
            with open(temp_file_path, 'r') as json_file:
                self.map = json.load(json_file)
            os.remove(temp_file_path)
            if self.verbose:
                print("mapping connected")
        except: # it doesn't exist
            if self.cloud_manager.vault_exists(name_vecs(self.vault, self.user, self.api)): # but if the vault does
                self.map = {str(i): str(i) for i in range(self.vectors.get_n_items())}
                if self.verbose:
                    print("mapping 2 connected")
            else: # otherwise no map
                if self.verbose:
                    print("mapping does not exist")

    def add_to_map(self):
        self.map[str(self.x)] = str(uuid.uuid4())
        self.x +=1
    
    def load_vectors(self):
        start_time = time.time()
        temp_file_path = self.cloud_manager.download_to_temp_file(name_vecs(self.vault, self.user, self.api))
        self.vectors.load(temp_file_path)
        self.load_mapping()
        os.remove(temp_file_path)
        self.vecs_loaded = True
        if self.verbose:
            print("get load vectors --- %s seconds ---" % (time.time() - start_time))
    
    def reload_vectors(self):
        num_existing_items = self.vectors.get_n_items()
        new_index = get_vectors(self.dims)
        count = -1
        for i in range(num_existing_items):
            count += 1
            vector = self.vectors.get_item_vector(i)
            new_index.add_item(i, vector)
        self.x = count + 1
        self.vectors = new_index

    def upload_vectors(self):
        # upload the vectors
        vector_temp_file_path = None
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            vector_temp_file_path = temp_file.name
            self.vectors.save(vector_temp_file_path)
            byte = os.path.getsize(vector_temp_file_path)
            self.cloud_manager.upload_temp_file(vector_temp_file_path, name_vecs(self.vault, self.user, self.api, byte))

        if os.path.exists(vector_temp_file_path):
            os.remove(vector_temp_file_path)

        # upload the map
        map_temp_file_path = None
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            map_temp_file_path = temp_file.name
            json.dump(self.map, temp_file, indent=2)
        
        self.cloud_manager.upload_temp_file(map_temp_file_path, name_map(self.vault, self.user, self.api, byte))
        if os.path.exists(map_temp_file_path):
            os.remove(map_temp_file_path)
            
        self.items.clear()
        self.vectors = get_vectors(self.dims)
        self.x_checked = False
        self.vecs_loaded = False
        self.saved_already = False

    def split_text(self, text, min_threshold=1000, max_threshold=16000):
        '''
        Internal function
        Splits the given text into chunks of sentences such that each chunk's length 
        is at least min_threshold characters but does not exceed max_threshold characters.
        Sentences are not broken mid-way.
        '''
        segments = []
        sentence_spans = list(re.finditer(r"(?<=[.!?])\s+", text))
        
        current_segment = []
        current_length = 0
        sentence_start = 0

        for sentence_span in sentence_spans:
            sentence = text[sentence_start:sentence_span.end()]

            if current_length + len(sentence) > max_threshold:
                if current_segment:
                    segments.append(" ".join(current_segment))
                current_segment = [sentence]
                current_length = len(sentence)
            else:
                current_segment.append(sentence)
                current_length += len(sentence)

            if current_length >= min_threshold:
                segments.append(" ".join(current_segment))
                current_segment = []
                current_length = 0

            sentence_start = sentence_span.end()

        # Add the remaining sentences or partial sentences to the last segment
        last_sentence = text[sentence_start:]
        if last_sentence:
            current_segment.append(last_sentence)
        
        # Ensure that even the last segment is long enough if there are prior segments
        if current_segment and (current_length >= min_threshold or not segments):
            segments.append(" ".join(current_segment))

        if self.verbose:
            print(f'New text chunk of size: {len(current_segment)}') 
        
        return segments
    
    def get_items(self, ids: list = [], include_total=False) -> list:
        '''
            Get one or more items from the database. 
            Input the item id(s) in a list. -> Returns the items 

            - Example Single Item Usage:
                item = vault.get_items([132])

            - Example Multi-Item Usage:
                items = vault.get_items([132, 128, 393, 74, 644, 71])

            Sample return when called:
            `[{'data': 'sample_data__sample_data...', 
            'metadata': {'name': '', 'item_id': 1, 'created_at':
            '2023-07-16T04:29:00.730754', 'updated_at': '2023-07-16T04:29:00.730758'},
            'distance': 0.7101698517799377}]`
        '''
        results = []
        self.load_vectors()
        start_time = time.time()

        for i in ids:
            # Retrieve the item
            item_data = self.cloud_manager.download_text_from_cloud(cloud_name(self.vault, self.map[str(i)], self.user, self.api, item=True))
            # Retrieve the metadata
            meta_data = self.cloud_manager.download_text_from_cloud(cloud_name(self.vault, self.map[str(i)], self.user, self.api, meta=True))
            meta = json.loads(meta_data)
            build_return(results, item_data, meta)
            if self.verbose:
                print(f"Retrieved {len(ids)} items --- %s seconds ---" % (time.time() - start_time))

        if include_total:
            return [results, self.get_total_items()]
        else:
            return results


    def get_items_by_vector(self, vector, n: int = 4, include_distances=False):
        '''
            Internal function that returns vector similar items. Requires input vector, returns similar items
        '''
        try:
            self.load_vectors()
            start_time = time.time()
            if not include_distances:
                results = []
                vecs = self.vectors.get_nns_by_vector(vector, n)
                for vec in vecs:
                    # Retrieve the item
                    item_data = self.cloud_manager.download_text_from_cloud(cloud_name(self.vault, self.map[str(vec)], self.user, self.api, item=True))
                    # Retrieve the metadata
                    meta_data = self.cloud_manager.download_text_from_cloud(cloud_name(self.vault, self.map[str(vec)], self.user, self.api, meta=True))
                    meta = json.loads(meta_data)
                    build_return(results, item_data, meta)
                if self.verbose:
                    print(f"get {n} items back --- %s seconds ---" % (time.time() - start_time))
                return results
            else:
                results = []
                vecs, distances = self.vectors.get_nns_by_vector(vector, n, include_distances=include_distances)
                counter = 0
                for vec in vecs:
                    # Retrieve the item
                    item_data = self.cloud_manager.download_text_from_cloud(cloud_name(self.vault, self.map[str(vec)], self.user, self.api, item=True))
                    # Retrieve the metadata
                    meta_data = self.cloud_manager.download_text_from_cloud(cloud_name(self.vault, self.map[str(vec)], self.user, self.api, meta=True))
                    meta = json.loads(meta_data)
                    build_return(results, item_data, meta, distances[counter])
                    counter+=1
                if self.verbose:
                    print(f"Retrieved {n} items --- %s seconds ---" % (time.time() - start_time))
                return results
        except:
            return [{'data': 'No data has been added', 'metadata': {'no meta': 'No metadata has been added'}}]

    def get_similar_local(self, text, n: int = 4, include_distances=False, model="text-embedding-ada-002"):
        '''
            Returns similar items from the Vault as the one you entered, but locally
            (saves a few milliseconds and is sometimes used on production builds)
        '''
        vector = self.process_batch([text], never_stop=False, loop_timeout=180, model=model)[0]
        return self.get_items_by_vector(vector, n, include_distances=include_distances)
    
    def get_similar(self, text, n: int = 4, include_distances=False):
        '''
            Returns similar items from the Vault as the text you enter.
            Sample return when called:
            `[{'data': 'sample_data__sample_data...', 
            'metadata': {'name': '', 'item_id': 1, 'created_at':
            '2023-07-16T04:29:00.730754', 'updated_at': '2023-07-16T04:29:00.730758'},
            'distance': 0.7101698517799377}]`

            (Usually there are four items, but in this sample print out, we have removed the other 3 
            and shortened this one's data for brevity)

            This sample was called with "include_distances=True", which adds the "distance" field to the return.
            The distance can be useful for assessing similarity differences in the items returned. 
            Each item has its' own distance number.


        '''
        return call_get_similar(self.user, self.vault, self.api, self.openai_key, text, n, include_distances=include_distances, verbose=self.verbose)

    def add_item(self, text: str, meta: dict = None, name: str = None):
        """
            If your text length is greater than 15000 characters, you should use Vault.split_text(your_text) to 
            get a list of text segments that are the right size
        """
        self.check_index()
        new_item = itemize(self.vault, self.x, meta, text, name)
        self.items.append(new_item)
        self.add_to_map()

    def add(self, text: str, meta: dict = None, name: str = None, split=False, split_size=1000, max_threshold=16000):
        """
            If your text length is greater than 4000 tokens, Vault.split_text(your_text)  
            will automatically be added
        """

        if len(text) > 15000 or split:
            if self.verbose:
                print('Using the built-in "split_text()" function to get a list of texts') 
            texts = self.split_text(text, min_threshold=split_size, max_threshold=max_threshold) # returns list of text segments
        else:
            texts = [text]
        for text in texts:
            self.add_item(text, meta, name)

    def add_item_with_vector(self, text: str, vector: list, meta: dict = None, name: str = None):
        """
            If your text length is greater than 15000 characters, you should use Vault.split_text(your_text) to 
            get a list of text segments that are the right size
        """
        self.check_index()
        start_time = time.time()

        if self.ai.get_tokens(text) > 4000:
            raise 'Text length too long. Use the "split_text() function to get a list of text segments'

        # Add vector to vectorspace
        self.vectors.add_item(self.x, vector)
        self.items.append(itemize(self.vault, self.x, meta, text, name))
        self.add_to_map()

        if self.verbose:
            print("add item time --- %s seconds ---" % (time.time() - start_time))

    def process_batch(self, batch_text_chunks, never_stop, loop_timeout, model="text-embedding-ada-002"):
        '''
            Internal function
        '''
        loop_start_time = time.time()
        exceptions = 0
        while True:
            try:
                res = openai.embeddings.create(input=batch_text_chunks, model=model)
                break
            except Exception as e:
                last_exception_time = time.time()
                exceptions = 1 if time.time() - last_exception_time > 180 else + 1
                print(f"API Error: {e}. Sleeping {(exceptions * 5)} seconds")
                time.sleep((exceptions * 5))

                if not never_stop or (time.time() - loop_start_time) > loop_timeout:
                    try:
                        res = openai.embeddings.create(input=batch_text_chunks, model=model)
                        break
                    except Exception as e:
                        if exceptions >= 5:
                            print(f"API has failed for too long. Exiting loop with error: {e}.")
                            break
                        raise TimeoutError("Loop timed out")
        return [record.embedding for record in res.data]
        
    def get_vectors(self, batch_size: int = 32, never_stop: bool = False, loop_timeout: int = 777, model="text-embedding-ada-002"):
        '''
        Takes text data added to the vault, and gets vectors for them
        '''
        start_time = time.time()

        # If last_time isn't set, assume it's a very old time (e.g., 10 minutes ago)
        if not self.last_time:
            self.last_time = start_time - 600

        texts = [item['text'] for item in self.items]
        num_batches = int(np.ceil(len(texts) / batch_size))
        batches_text_chunks = [
            texts[i * batch_size:min((i + 1) * batch_size, len(texts))]
            for i in range(num_batches)
        ]

        batch_embeddings_list = [self.process_batch(batch_text_chunk, never_stop=never_stop, loop_timeout=loop_timeout, model=model) for batch_text_chunk in batches_text_chunks]

        current_item_index = 0
        for batch_embeddings in batch_embeddings_list:
            for embedding in batch_embeddings:
                item_index = self.items[current_item_index]["meta"]["item_id"]
                self.vectors.add_item(item_index, embedding)
                current_item_index += 1

        self.last_time = time.time()
        if self.verbose:
            print("get vectors time --- %s seconds ---" % (time.time() - start_time))


    def get_chat(self, text: str = None, history: str = None, summary: bool = False, get_context = False, n_context = 4, return_context = False, history_search = False, model='gpt-3.5-turbo', include_context_meta=False, custom_prompt=False, local=False, temperature=0):
        '''
            Chat get response from OpenAI's ChatGPT. 
            Models: ChatGPT = "gpt-3.5-turbo" • GPT4 = "gpt-4" 
            Large Context Models: ChatGPT 16k = "gpt-3.5-turbo-16k" • GPT4 32k = "gpt-4-32k"
            Best Versions: "gpt-3.5-turbo-0301" is March 2023 ChatGPT (best version) - "gpt-4-0314" (best version)

            Rate limiting, auto retries, and chat histroy slicing built-in so you can chat with ease. 
            Enter your text, add optional chat history, and optionally choose a summary response (default: summmary = False)

            - Example Signle Usage: 
            `response = vault.get_chat(text)`

            - Example Chat: 
            `response = vault.get_chat(text, chat_history)`
            
            - Example Summary: 
            `summary = vault.get_chat(text, summary=True)`

            - Example Context-Based Response:
            `response = vault.get_chat(text, get_context = True)`

            - Example Context-Based Response w/ Chat History:
            `response = vault.get_chat(text, chat_history, get_context = True)`

            - Example Context-Response with Context Samples Returned:
            `vault_response = vault.get_chat(text, get_context = True, return_context = True)`
            
            Response is a string, unless return_context == True, then response will be a dictionary 

            - Example to print dictionary results:
            # print response:
            `print(vault_response['response'])` 

            # print context:
            for item in answer['context']:
                print("\n\n", f"item {item['metadata']['item_id']}")
                print(item['data'])

            history_search is False by default skip adding the history of the conversation to the text input for similarity search (useful if history contains subject infomation useful for answering the new text input and the text input doesn't contain that info)
            
            - Example Custom Prompt:
            `response = vault.get_chat(text, chat_history, get_context=True, custom_prompt=my_prompt)`

            `custom_prompt` overrides the stock prompt we provide. Check ai.py to see the originals we provide. 
            `llm` and `llm_stream` models manage history internally, so the `content` is the only variable to be included and formattable in the prompt. 

            *Example WIHTOUT Vault Context:*

            ```python
            my_prompt = """Answer this question as if you were a financial advisor: "{content}". """
            response = vault.get_chat(text, chat_history, get_context=True, custom_prompt=my_prompt)
            ```

            Getting context from the Vault is usually the goal when customizing text generation, and doing that requires additional prompt variables.
            `llm_w_context` and `llm__w_context_stream` models inject the history, context, and user input all in one prompt. In this case, your custom prompt needs to have `history`, `context` and `question` formattable in the prompt like so:

            *Example WITH Vault Context:*  
            ```python
            custom_prompt = """
                Use the following Context to answer the Question at the end. 
                Answer as if you were the modern voice of the context, without referencing the context or mentioning that fact any context has been given. Make sure to not just repeat what is referenced. Don't preface or give any warnings at the end.

                Chat History (if any): {history}

                Additional Context: {context}

                Question: {question}

                (Respond to the Question directly. Be the voice of the context, and most importantly: be interesting, engaging, and helpful) 
                Answer:
            """ 
            response = vault.get_chat(text, chat_history, get_context=True, custom_prompt=my_prompt)
            ```

        '''
        model = model.lower()
        start_time = time.time()
        if not history:
            history = ''

        if text: 
            if not self.ai.within_context_window(text, model):
                if summary:
                    inputs = self.split_text(text, self.ai.model_token_limits.get(model, 15000) * 3)
                else:
                    inputs = text[-(self.ai.model_token_limits.get(model, 15000) * 3):]
            else:
                inputs = [text]
        else:
            if not custom_prompt:
                raise ValueError("No input text provided. Please enter text to proceed.")
            else:
                inputs = []

        response = ''
        for segment in inputs:
            attempts = 0
            while True:
                try:
                    # Make your API call here
                    if summary and not get_context:
                        response += self.ai.summarize(segment, model=model, custom_prompt=custom_prompt, temperature=temperature)
                    elif text and get_context and not summary:
                        user_input = segment + history if history_search else segment
                        if include_context_meta:
                            context = self.get_similar(user_input, n=n_context) if not local else self.get_similar_local(user_input, n=n_context)
                            input_ = str(context)
                        else:
                            context = self.get_similar(user_input, n=n_context) if not local else self.get_similar_local(user_input, n=n_context)
                            input_ = ''
                            for text in context:
                                input_ += text['data']
                        response = self.ai.llm_w_context(segment, input_, history, model=model, custom_prompt=custom_prompt, temperature=temperature)
                    else: # Custom prompt only
                        response = self.ai.llm(segment, history, model=model, custom_prompt=custom_prompt, temperature=temperature)

                    # If the call is successful, reset the backoff
                    self.rate_limiter.on_success()
                    break
                except Exception as e:
                    # If the call fails, apply the backoff
                    attempts += 1
                    print(traceback.format_exc())
                    print(f"API Error: {e}. Backing off for {self.rate_limiter.current_delay} seconds.")
                    self.rate_limiter.on_failure()
                    if attempts >= self.rate_limiter.max_attempts:
                        print(f"API Failed too many times, exiting loop: {e}.")
                        break

        if self.verbose:
            print("get chat time --- %s seconds ---" % (time.time() - start_time))

        if not return_context:
            return response
        elif return_context:
            return {'response': response, 'context': context}
        
    def get_chat_stream(self, text: str = None, history: str = None, summary: bool = False, get_context = False, n_context = 4, return_context = False, history_search = False, model='gpt-3.5-turbo', include_context_meta=False, metatag=False, metatag_prefixes=False, metatag_suffixes=False, custom_prompt=False, local=False, temperature=0):
        '''
            Always use this get_chat_stream() wrapped by either print_stream(), or cloud_stream().
            cloud_stream() is for cloud functions, like a flask app serving a front end elsewhere.
            print_stream() is for local console printing

            - Example Signle Usage: 
            `response = vault.print_stream(vault.get_chat_stream(text))`

            - Example Chat: 
            `response = vault.print_stream(vault.get_chat_stream(text, chat_history))`
            
            - Example Summary: 
            `summary = vault.print_stream(vault.get_chat_stream(text, summary=True))`

            - Example Context-Based Response:
            `response = vault.print_stream(vault.get_chat_stream(text, get_context = True))`

            - Example Context-Based Response w/ Chat History:
            `response = vault.print_stream(vault.get_chat_stream(text, chat_history, get_context = True))`

            - Example Context-Response with Context Samples Returned:
            `vault_response = vault.print_stream(vault.get_chat_stream(text, get_context = True, return_context = True))`

            - Example Context-Response with SPECIFIC META TAGS for Context Samples Returned:
            `vault_response = vault.print_stream(vault.get_chat_stream(text, get_context = True, return_context = True, include_context_meta=True, metatag=['title', 'author']))`
            
            - Example Context-Response with SPECIFIC META TAGS for Context Samples Returned & Specific Meta Prefixes and Suffixes:
            `vault_response = vault.print_stream(vault.get_chat_stream(text, get_context = True, return_context = True, include_context_meta=True, metatag=['title', 'author'], metatag_prefixes=['\n\n Title: ', '\nAuthor: '], metatag_suffixes=['', '\n']))`
            

            - Example Custom Prompt:
            `response = vault.get_chat(text, chat_history, get_context=True, custom_prompt=my_prompt)`

            `custom_prompt` overrides the stock prompt we provide. Check ai.py to see the originals we provide. 
            `llm` and `llm_stream` models manage history internally, so the `content` is the only variable to be included and formattable in the prompt. 

            *Example WIHTOUT Vault Context:*

            ```python
            my_prompt = """Answer this question as if you were a financial advisor: "{content}". """
            response = vault.print_stream(vault.get_chat_stream(text, chat_history, get_context = True, custom_prompt=my_prompt))
            ```

            Getting context from the Vault is usually the goal when customizing text generation, and doing that requires additional prompt variables.
            `llm_w_context` and `llm__w_context_stream` models inject the history, context, and user input all in one prompt. In this case, your custom prompt needs to have `history`, `context` and `question` formattable in the prompt like so:

            *Example WITH Vault Context:*  
            ```python
            custom_prompt = """
                Use the following Context to answer the Question at the end. 
                Answer as if you were the modern voice of the context, without referencing the context or mentioning that fact any context has been given. Make sure to not just repeat what is referenced. Don't preface or give any warnings at the end.

                Chat History (if any): {history}

                Additional Context: {context}

                Question: {question}

                (Respond to the Question directly. Be the voice of the context, and most importantly: be interesting, engaging, and helpful) 
                Answer:
            """ 
            response = vault.print_stream(vault.get_chat_stream(text, chat_history, get_context = True, custom_prompt=my_prompt))
            ```
        '''

        model = model.lower()
        start_time = time.time()
        if not history:
            history = ''

        if text:
            if not self.ai.within_context_window(text, model):
                if summary:
                    inputs = self.split_text(text, self.ai.model_token_limits.get(model, 15000) * 3)
                else:
                    inputs = text[-(self.ai.model_token_limits.get(model, 15000) * 3):]
            else:
                inputs = [text]
        else:
            if not custom_prompt:
                raise ValueError("No input text provided. Please enter text to proceed.")
            else:
                inputs = []
                
        counter = 0
        for segment in inputs:
            if self.verbose:
                print(f"segments: {len(inputs)}")
            start_time = time.time()
            exceptions = 0

            while True:
                try:
                    if summary and not get_context:
                        try:
                            for word in self.ai.summarize_stream(segment, model=model, custom_prompt=custom_prompt, temperature=temperature):
                                yield word
                            self.rate_limiter.on_success()
                        except Exception as e:
                            raise e
                        if counter == len(inputs):
                            yield '!END'
                            self.rate_limiter.on_success()
                    
                    elif text and get_context and not summary:
                        user_input = segment + history 

                        context = self.get_similar(user_input, n=n_context) if not local else self.get_similar_local(user_input, n=n_context)
                        input_ = str(context) if include_context_meta else ''
                        for text in context:
                            input_ += text['data']

                        try:
                            for word in self.ai.llm_w_context_stream(segment, input_, history, model=model, custom_prompt=custom_prompt, temperature=temperature):
                                yield word
                            self.rate_limiter.on_success()
                        except Exception as e:
                            raise e

                        if return_context:
                            for item in context:
                                if not metatag:
                                    for tag in item['metadata']:
                                        yield str(item['metadata'][f'{tag}'])
                                else:
                                    if metatag_prefixes:
                                        if metatag_suffixes:
                                            for i in range(len(metatag)):
                                                yield str(metatag_prefixes[i]) + str(item['metadata'][f'{metatag[i]}']) + str(metatag_suffixes[i])
                                        else:
                                            for i in range(len(metatag)):
                                                yield str(metatag_prefixes[i]) + str(item['metadata'][f'{metatag[i]}'])
                                yield item['data']
                            yield '!END'
                            self.rate_limiter.on_success()
                        else:
                            yield '!END'
                            self.rate_limiter.on_success()

                    else:
                        try:
                            for word in self.ai.llm_stream(segment, history, model=model, custom_prompt=custom_prompt, temperature=temperature):
                                yield word
                            self.rate_limiter.on_success()
                        except Exception as e:
                            raise e
                        yield '!END'
                    self.rate_limiter.on_success()
                    break
                except Exception as e:
                    exceptions += 1
                    print(f"API Error: {e}. Applying backoff.")
                    self.rate_limiter.on_failure()
                    if exceptions >= self.rate_limiter.max_attempts:
                        print(f"API Failed too many times, exiting loop: {e}.")
                        break

        if self.verbose:
            print("get chat time --- %s seconds ---" % (time.time() - start_time))

    def print_stream(self, function, printing=True):
        '''
            For local use printing the chat stream. Call 'printing=False' for no pretty printing to be applied
        '''
        full_text= ''
        newlinetime=1
        for word in function:
            if word != '!END' and word:
                full_text += word
                if printing:
                    if len(full_text) / 80 > newlinetime:
                        newlinetime += 1
                        print(f'\n{word}', end='', flush=True)
                    else:
                        print(word, end='', flush=True) 
            else:
                return full_text
    
    def cloud_stream(self, function):
        '''
            For cloud application yielding the chat stream, like a flask app
        '''
        for word in function:
            yield f"data: {json.dumps({'data': word})} \n\n"

class RateLimiter:
    def __init__(self, max_attempts=30):
        self.base_delay = 1  # Base delay of 1 second
        self.max_delay = 60  # Maximum delay of 60 seconds
        self.backoff_factor = 2
        self.current_delay = self.base_delay
        self.max_attempts = max_attempts

    def on_success(self):
        # Reset delay after a successful call
        self.current_delay = self.base_delay

    def on_failure(self):
        # Apply exponential backoff with a random jitter
        self.current_delay = min(self.max_delay, random.uniform(self.base_delay, self.current_delay * self.backoff_factor))
        time.sleep(self.current_delay)