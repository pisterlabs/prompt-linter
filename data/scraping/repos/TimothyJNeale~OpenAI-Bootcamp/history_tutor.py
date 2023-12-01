# Use ChatGTP API to create a history tutor
######################################### IMPORTS #############################################
import openai
import logging
import os

import tiktoken

from dotenv import load_dotenv

######################################## CONSTANTS ############################################

DATA_DIRECTORY ='data'
GPT_SYSTEM_ROLE = "You are a USA history teacher for middle school kids."

########################################### DATA ##############################################

# load environment variables from .env file
load_dotenv()

##################################### HELPER FUCTIONS #########################################

# Standard completion
def get_completion(prompt, model="gpt-3.5-turbo-instruct", temperature=0, max_tokens=300, stop="\"\"\""):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop)

    return response.choices[0].text

# Get the number of tokens in a string
def get_num_tokens_from_string(string, encoding_name="gpt2"):
    tokenizer = tiktoken.get_encoding(encoding_name)
    tokens = tokenizer.encode(string)

    return len(tokens)

######################################## LOGGING ##############################################
logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')
logging.disable(logging.DEBUG) # Supress debugging output from modules imported
#logging.disable(logging.CRITICAL) # Uncomment to disable all logging

######################################## CLASSES ##############################################
class CreateBot:

    def __init__(self, model="gpt-3.5-turbo", system_prompt=GPT_SYSTEM_ROLE, temperature=0, max_tokens=150):
        self.system = system_prompt
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.messages = [{"role": "system", "content": system_prompt}]

    def chat(self):
        print('To terminate the chat, type "quit"')
        question = ''

        while question.lower() != 'quit':
            question = input('You: ')
            if question.lower() == 'quit':
                break
            self.messages.append({"role": "user", "content": question})
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=self.messages,
                temperature=0,
                max_tokens=150
                )
            
            answer = response.choices[0].message["content"]
            self.messages.append({"role": "assistant", "content": answer})
            print('Bot: ' + answer)


######################################### START ###############################################
logging.info('Start of program')

# Get the current DATA directory
home = os.getcwd()
data_dir = os.path.join(home, DATA_DIRECTORY)
logging.info(data_dir)

os.chdir(data_dir)

# Authenticate with OpenAI                             
api_key = os.environ["OPENAI_API_KEY"]
openai.api_key = api_key

########################################## MAIN ###############################################
logging.info('Main section entered')

# # Use ChatGTP API to create a history tutor
history_tutor = CreateBot()
history_tutor.chat()

######################################### FINISH ##############################################
logging.info('End of program')