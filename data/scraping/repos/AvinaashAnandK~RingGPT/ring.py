import logging
from itertools import cycle
import pandas as pd
from Bard import Chatbot as BardChatbot
import asyncio
from EdgeGPT import Chatbot as EdgeChatbot
from revChatGPT.V1 import Chatbot as RevChatbot
import requests
import time
from rich import print
import random
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from .dataloader import DataLoader
from .prompt import Prompt

class Ring:
    def __init__(self, llm_list, max_attempts=5, retry_delay=5, throttle_delay=5, log_enabled=True, wip_save=True, prompt=None, data_loader = None):
        # Validate the llm_list
        if not llm_list:
            raise ValueError("Failed initializing Ring, llm_list is not declared by the user")
        if not prompt:
            raise ValueError("Failed initializing Ring, prompt is not declared by the user")
        if not data_loader:
            raise ValueError("Failed initializing Ring, data_loader is not declared by the user")
        if not isinstance(llm_list, list) or not all(isinstance(llm, dict) for llm in llm_list):
            raise ValueError("""Failed initializing Ring, llm_list does not align with the expected format. Please pass llm_list in the format:
                [{"service": "OpenAIChat", 
                  "access_token": "enter-your-token-here",
                  "identifier": "identifier-for-the-account-used-for-the-token"},
                 {"service": "BardChat", 
                  "access_token": "enter-your-token-here",
                  "identifier": "identifier-for-the-account-used-for-the-token"},
                 {"service": "EdgeChat"}]""")
        for llm in llm_list:
            if 'service' not in llm:
                raise ValueError("Failed initializing Ring, service attribute is not declared by the user in the dictionaries passed in llm_list")
            if llm['service'] in ["OpenAIChat", "BardChat"] and (not llm.get('access_token')):
                raise ValueError("Failed initializing Ring, access_token attribute is not declared by the user in the dictionaries passed in llm_list for OpenAIChat or BardChat")
        
        self.data_loader = data_loader
        data_loader.load_data()
        
        self.prompt = prompt
        
        self.llm_list = llm_list
        
        self.max_attempts = max_attempts
        self.retry_delay = retry_delay
        self.throttle_delay = throttle_delay
        self.log_enabled = log_enabled
        self.wip_save = wip_save

        self.output_dataframe = self.data_loader.get_data()
        

        # Set up logging
        if self.log_enabled:
            logging.basicConfig(filename='ring_log.log', filemode='a', level=logging.INFO, 
                                format='%(asctime)s - %(levelname)s - %(message)s')
            logging.info("Ring initialized")

        self.llm_iterator = cycle(llm_list)
        
        self.tokenizer = tiktoken.get_encoding('cl100k_base')

        self.instruction_prompt = self.prompt.get_instruction_prompt() + self.example_picker(self.prompt.get_example())

        self.data_column_name = self.data_loader.get_data_column_name()
        self.process_data(data_column_name=self.data_column_name) 

        self.instruction_categories = self.prompt.get_instruction_categories()

    def instruction_categories_picker(self, instruction_categories):
        if len(instruction_categories) == 1:
            return instruction_categories[0]
        elif len(instruction_categories) > 1:
            return random.sample(instruction_categories, 2)
        else:
            return None

    def example_picker(self, example):
        if not example:
            return ""

        if len(example) == 1:
            return f"### EXAMPLE ###\nContext: {example[0]['input']}\nOutput: {example[0]['output']}"

        selected_example = example[0]
        unselected_examples = example[1:]
        if len(unselected_examples) > 0:
            selected_example_index = random.randint(0, len(unselected_examples) - 1)
            selected_example = unselected_examples.pop(selected_example_index)

        output = f"### EXAMPLES ###\n\nExample 1\nContext: {selected_example['input']}\nOutput: {selected_example['output']}\n\n"

        if unselected_examples:
            output += "Example 2\n"
            second_example = random.choice(unselected_examples)
            output += f"Context: {second_example['input']}\nOutput: {second_example['output']}"

        return output

    def tokenizer_len(self, text):
        tokens = self.tokenizer.encode(
            text,
            disallowed_special=()
        )
        return len(tokens)

    def process_data(self, data_column_name=None):
        prompt_length = self.tokenizer_len(self.instruction_prompt)
        chunk_size = 2000 - prompt_length

        self.text_chunker = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=20,
        length_function=self.tokenizer_len,
        separators=['\n\n', '\n', ' ', '']
        )

        self.output_dataframe['chunk'] = self.output_dataframe[data_column_name].apply(lambda x: self.text_chunker.split_text(x))
        self.output_dataframe = self.output_dataframe.explode('chunk')
        self.output_dataframe.reset_index(drop=True, inplace=True)

        # Repeat other columns for each row
        self.output_dataframe = self.output_dataframe.reindex(self.output_dataframe.index.repeat(len(self.output_dataframe.columns) - 1)).reset_index(drop=True)
        self.output_dataframe[data_column_name] = self.output_dataframe['chunk']
        self.output_dataframe.drop(columns=['chunk'], inplace=True)

    def BardChat(self, prompt, token):
        logging.info("Function BardChat called")
        try:
            chatbot = BardChatbot(token)
            response = chatbot.ask(prompt)
            response_raw = {
                "service": "BardChat",
                "response": response,
                "response_clean": response['content']
            }
            return response_raw
        except Exception as e:
            logging.error(f"Exception occurred with error: {e}")
            response_raw = {
                "service": "BardChat",
                "response": "Error",
                "response_clean": "Error"
            }
            return response_raw

    def OpenAIChat(self, prompt, token):
        logging.info("Function OpenAIChat called")
        try:
            chatbot = RevChatbot(config={"access_token": f"{token}"})
            for data in chatbot.ask(prompt):
                response = data

            response_raw = {
                "service": "OpenAIChat",
                "response": response,
                "response_clean": response["message"]
            }

        except Exception as e:
            logging.error(f"Exception occurred with error: {e}")
            response_raw = {
                "service": "OpenAIChat",
                "response": "Error",
                "response_clean": "Error"
            }
            return response_raw

        return response_raw

    async def EdgeChat(self, prompt, token):
        try:
            logging.info("Function EdgeChat called")
            chatbot = EdgeChatbot()
            response = await chatbot.ask(prompt=prompt)
            response_raw = {
                "service": "EdgeChat",
                "response": response,
                "response_clean": response['item']['messages'][1]['text']
            }
            return response_raw
        except Exception as e:
            logging.error(f"Exception occurred with error: {e}")
            response_raw = {
                "service": "EdgeChat",
                "response": "Error",
                "response_clean": "Error"
            }
            return response_raw
        
    async def run(self):
        for i, row in self.dataframe.iterrows():
            text = row['text_final']
            if self.data_column_name:
                prompt = self.instruction_prompt + row[self.data_column_name]
            else:
                prompt = self.instruction_prompt + row[0]

            check_1, check_2 = self.instruction_categories_picker(self.instruction_categories)
            
            logging.info(f"Row {i} to be parsed")

            for attempt in range(self.max_attempts):
                token = next(self.llm_iterator)

            try:
                if token['service'] == "EdgeChat":
                    response = await self.EdgeChat(prompt, token['access_token'])
                elif token['service'] == "OpenAIChat":
                    response = self.OpenAIChat(prompt, token['access_token'])
                elif token['service'] == "BardChat":
                    response = self.BardChat(prompt, token['access_token'])


                # Check response status code and handle accordingly
                if check_1 in response['response_clean'].lower() and check_2 in response['response_clean'].lower():
                    # Process the response if everything's OK
                    logging.info(f"Response received for row: {i} - from: {response['service']}")

                    # Extract values from the response dictionary
                    service = response['service']
                    response_val = response['response']
                    response_clean = response['response_clean']

                    # Store values in the DataFrame
                    self.output_dataframe.loc[i, 'service'] = str(service)
                    self.output_dataframe.loc[i, 'response'] = str(response_val)
                    self.output_dataframe.loc[i, 'response_clean'] = str(response_clean)

                    # Stop the retry loop if the request was successful
                    break
                else:
                    # Handle different status codes and possibly use a different API key if the limit is reached
                    logging.warning(f"Request failed for row: {i} - from: {response['service']}. Attempt {attempt + 1} of {self.max_attempts}")
                    # Wait before the next retry
                    time.sleep(self.retry_delay)
            except requests.exceptions.RequestException as e:
                logging.error(f"Exception occurred with error: {e}")
                # Wait before the next retry
                time.sleep(self.retry_delay)

        # Throttle requests by waiting after each one
            if self.wip_save:
                self.output_dataframe.dropna(subset=['service', 'response', 'response_clean'], inplace=True)
                self.output_dataframe.to_csv("output.csv", index=False)

            time.sleep(self.throttle_delay)
        
        print("Done!")
        print(self.output_dataframe)
