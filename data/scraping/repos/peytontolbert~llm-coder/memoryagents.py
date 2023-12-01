import openai
from gptfunctions import ChatGPTAgent
from dotenv import load_dotenv
import os
# Initialize OpenAI and GitHub API keys
openai.api_key = os.getenv('OPENAI_API_KEY')
def memory_consolidation_agent(memory):
    prompt = "You are an AI that specializes in memory consolidation. \n\nYour task is to consolidate the provided memories into a single memory. \n\nReturn the consolidated memory as a Python dictionary."
    print(prompt)
    result = ChatGPTAgent.chat_with_gpt3(prompt, memory)

def contextual_understanding_agent(memories):
    prompt = "You are an AI that specializes in contextual understanding. \n\nYour task is to understand the context of the provided memories. \n\nReturn the context as a Python dictionary."
    print(prompt)
    result = ChatGPTAgent.chat_with_gpt3(prompt, memories)

def memory_classification_agent(context):
    prompt = "You are an AI that specializes in memory classification. \n\nYour task is to classify the provided memories. \n\nReturn the classification as a Python dictionary."
    print(prompt)
    result = ChatGPTAgent.chat_with_gpt3(prompt, context)

def memory_compression_agent(context):
    prompt = "You are an AI that specializes in memory compression. \n\nYour task is to compress the provided memory. \n\nReturn the compressed memory as a Python dictionary."
    print(prompt)
    result = ChatGPTAgent.chat_with_gpt3(prompt, context)

def memory_retrieval_agent(context):
    prompt = "You are an AI that specializes in memory retrieval. \n\nYour task is to retrieve the memories that match the provided context. \n\nReturn the retrieved memories as a Python dictionary."
    print(prompt)
    result = ChatGPTAgent.chat_with_gpt3(prompt, context)

def memory_validation_agent(memory):
    prompt = "You are an AI that specializes in memory validation. \n\nYour task is to validate the provided memory. \n\nReturn the validation result as a Python dictionary."
    print(prompt)
    result = ChatGPTAgent.chat_with_gpt3(prompt, memory)
