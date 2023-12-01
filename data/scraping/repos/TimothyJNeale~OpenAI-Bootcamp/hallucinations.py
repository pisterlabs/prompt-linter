# Examples of hallucinations and how to avoid them 

######################################### IMPORTS #############################################
import openai
import logging
import os

from dotenv import load_dotenv

######################################## CONSTANTS ############################################


DATA_DIRECTORY ='data'


########################################### DATA ##############################################

# load environment variables from .env file
load_dotenv()


##################################### HELPER FUCTIONS #########################################

# Use chat completion
def get_chat_completion(prompt, model="gpt-3.5-turbo", temperature=0):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    return response.choices[0].message["content"]

# Standard completion
def get_completion(prompt, model="gpt-3.5-turbo-instruct", temperature=0, max_tokens=300, stop="\"\"\""):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop)

    return response.choices[0].text

######################################## LOGGING ##############################################

logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')
logging.disable(logging.DEBUG) # Supress debugging output from modules imported
#logging.disable(logging.CRITICAL) # Uncomment to disable all logging


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

prompt = '''What does the start-up Super Dooper Pooper Scooper do and who are the investors and how much seed funding have they provided?'''

response = get_completion(prompt, model="gpt-3.5-turbo-instruct", temperature=0, max_tokens=512)
logging.info(f"Prompt 1 : {response}\n")


prompt = '''Q: What does the start-up Super Dooper Pooper Scooper do and who are the investors and how much seed funding have they provided?
            A: '''

response = get_completion(prompt, model="gpt-3.5-turbo-instruct", temperature=0, max_tokens=512)
logging.info(f"Prompt 2 : {response}\n")

prompt = '''Only answer the question below if you are 100% certain of the facts. If you are not certain, please leave the answer blank.

Q: What does the start-up Super Dooper Pooper Scooper do and who are the investors and how much seed funding have they provided?
A: '''

response = get_completion(prompt, model="gpt-3.5-turbo-instruct", temperature=0, max_tokens=512)
logging.info(f"Prompt 3 : {response}\n")

######################################### FINISH ##############################################
logging.info('End of program')