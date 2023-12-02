# from gpt4all import GPT4All
from langchain.llms import OpenAI, GPT4All
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import CommaSeparatedListOutputParser
import csv
import pandas as pd
from dotenv import load_dotenv
import os
import json
import logging

load_dotenv(dotenv_path="../../.env")

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# model = OpenAI(os.getenv("OPENAI_API_KEY"))
# llm = GPT4All(model="orca-mini-3b.ggmlv3.q4_0.bin", n_threads=8)

output_parser = CommaSeparatedListOutputParser()

format_instructions = output_parser.get_format_instructions()

def generateTopicList(topic, location, list_length=10, save=False, path=None):
    list_length = str(list_length)

    prompt = PromptTemplate(
            input_variables=["topic", "location", "list_length"],
            template="List the top {list_length} most popular {topic} in the {location}?.\n{format_instructions}",
            partial_variables={"format_instructions": format_instructions}

        )
    llm = OpenAI(model="text-davinci-003" , temperature=0.0)

    chain = LLMChain(llm=llm, prompt=prompt)

    topic_list = output_parser.parse(
    chain.run({
        'topic': topic,
        'location': location,
        'list_length': list_length
        }))


    
    topic_df = pd.DataFrame({topic: topic_list})

    if save:

        # Create the directories if they don't exist
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        topic_df.to_csv(f"{path}{topic}_{location}_list.csv", index=True, header=True, index_label="Index")

    return topic_df

# generateTopicList("sports", "United States", 10, save=True)