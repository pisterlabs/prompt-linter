# Chains - Combining different LLM calls and actions automatically

# Example: Summary #1, Summary #2, Summary #3 > Final Summary
# Check out the video exploring different summarisation chain types
# There are many different applications of chains; search to see which are best for your use case.
# We'll cover two of them...
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# 1. Simple Sequential Chains
# Easy chains where you can the use output of an LLM as an input into another. Good for breaking up tasks and keeping the LLM focused
# from langchain.llms import OpenAI
# from langchain.chains import LLMChain 
# from langchain.prompts import PromptTemplate
# from langchain.chains import SimpleSequentialChain

# llm = OpenAI(temperature=1, openai_api_key=openai_api_key)

# template = """
# Your job is to come up with a classic dish from the area that the user suggests.
# % USER LOCATION
# {user_location}

# YOUR RESPONSE:
# """

# prompt_template = PromptTemplate(input_variables=["user_location"], template=template)

# # Holds my 'location' chain
# location_chain = LLMChain(llm=llm, prompt=prompt_template)

# template = """
# Given a meal, give a short and simple recipe on how to make that dish at home
# % MEAL
# {user_meal}

# YOUR RESPONSE:
# """
# prompt_template = PromptTemplate(input_variables=["user_meal"], template=template)

# # Holds my 'meal' chain
# meal_chain = LLMChain(llm=llm, prompt=prompt_template)

# overall_chain = SimpleSequentialChain(chains=[location_chain, meal_chain], verbose=True)
# review = overall_chain.run("Rome")


# 2. Summarisation Chain
# Easily run through long numerous documents and get a summary. 
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = TextLoader('data/pgessay.txt')
documents = loader.load()
llm = OpenAI(temperature=1, openai_api_key=openai_api_key)

# Get your splitter ready
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)

# Split your docs into texts
texts = text_splitter.split_documents(documents)

# There is a lot of complexity hidden in this one line. Check out the docs.
chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
print(chain.run(texts[0:2]))