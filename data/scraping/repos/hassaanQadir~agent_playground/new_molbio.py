# Standard library imports
import asyncio
import itertools
import json
import os
import time

# Third party imports
import openai
import pinecone

# Local application imports
from dotenv import load_dotenv
from langchain import LLMChain, PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Load environment variables from '.env' file
load_dotenv('.env')

# Get the OpenAI and Pinecone API keys from environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')

# Load agents' data from JSON file 'agents.json'
with open('agents.json', 'r') as f:
    agentsData = json.load(f)

# Set the OpenAI and Pinecone API keys for use in their respective libraries
openai.api_key = openai_api_key
pinecone.init(api_key=pinecone_api_key, environment="us-west1-gcp")

# Pinecone index for vectorizing OpenTrons API
index_name = 'opentronsapi-docs'
index = pinecone.Index(index_name)

# Function definitions for main program logic
def create_llmchain(agent_id):
    """
    Create a LLMChain for a specific agent by calling on prompts stored in agents.json

    :param agent_id: The ID of the agent
    :return: An instance of LLMChain
    """
    chat = ChatOpenAI(streaming=False, callbacks=[StreamingStdOutCallbackHandler()], temperature=0, openai_api_key=openai_api_key)
    template = agentsData[agent_id]['agent{}_template'.format(agent_id)]
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    example_human = HumanMessagePromptTemplate.from_template(agentsData[agent_id]['agent{}_example1_human'.format(agent_id)])
    example_ai = AIMessagePromptTemplate.from_template(agentsData[agent_id]['agent{}_example1_AI'.format(agent_id)])
    human_message_prompt = HumanMessagePromptTemplate.from_template("{text}")
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, example_human, example_ai, human_message_prompt])
    return LLMChain(llm=chat, prompt=chat_prompt)

async def async_generate(chain, prompt):
    """
    Asynchronous function to generate a response with the specified chain and prompt

    :param chain: The chain to use for response generation
    :param prompt: The prompt to provide to the chain
    :return: The generated response
    """
    resp = await chain.arun({'text': prompt})
    return resp

async def generate_concurrently(chain, inputList):
    """
    Asynchronous function to generate responses concurrently for a list of inputs

    :param chain: The chain to use for response generation
    :param inputList: The list of prompts to provide to the chain
    :return: A list of generated responses
    """
    tasks = [async_generate(chain, input) for input in inputList]
    outputList = await asyncio.gather(*tasks)
    return outputList

def process_results(rawList):
    """
    Process raw list of generated responses to clean and consolidate results

    :param rawList: The raw list of generated responses
    :return: A list of cleaned and consolidated responses
    """
    allCleanedItems = [s for item in rawList for s in item.split('|||') if len(s) >= 10]
    return allCleanedItems

def applyLayer(chain, inputList):
    """
    Apply a layer of response generation to a list of inputs

    :param chain: The chain to use for response generation
    :param inputList: The list of prompts to provide to the chain
    :return: A list of generated responses
    """
    rawList = asyncio.run(generate_concurrently(chain, inputList))
    finalList = process_results(rawList)
    return finalList

def displayOutput(list1, list2, list3):
    """
    Display the generated responses in a nested dictionary format

    :param list1: The list of responses from the first chain
    :param list2: The list of responses from the second chain
    :param list3: The list of responses from the third chain
    :return: A nested dictionary of generated responses
    """
    nested_dict = {}
    for l1 in list1:
        nested_dict[l1] = {}
        for l2 in list2[:3]:
            nested_dict[l1][l2] = {}
            for l3 in list3[:8]:
                nested_dict[l1][l2][l3] = None
            del list3[:8]
        del list2[:3]
    return nested_dict

def driver(user_input):
    """
    Main function to drive the sequence of operations for the response generation process

    :param user_input: The initial user input to provide to the first chain
    :return: A nested dictionary of generated responses
    """
    user_input = [user_input]
    outputData = {}

    print("Entered driver function")

    # Create chains
    chains = [create_llmchain(i) for i in range(1, 4)]
    print("Chains created")
    
    # Apply layers of response generation
    layers = []
    for i, chain in enumerate(chains):
        layer = applyLayer(chain, user_input if i == 0 else layers[-1])
        layers.append(layer)
        
    print("Completed all layers")
    
    # Organize output in a nested dictionary
    outputData = displayOutput(*layers)
    print("Created nested dictionary")

    return outputData

def test(user_input):
    time.sleep(30)
    user_input += "Successfully accessed\n"
    user_input += "the molbio.ai"
    return user_input

if __name__ == "__main__":
   # Main entry point for the script
   answer = driver("Make glow in the dark e. coli")
   with open('answer.json', 'w') as f:
       # Write the answer to 'answer.json' file
       json.dump(answer, f, indent=4)