import config
import os
import json
os.environ['OPENAI_API_KEY'] = config.openAI
import openai

from langchain import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
# text = "Hello!! Can you help me with my marketing and sales tasks?"
# template="""
#     You are a helpful marketing and sales assistant that asks me for your age, previous companies 
#     you worked in and the projects that you lead. Then you are going to induce that personality into 
#     you and you are going to help me with my marketing and sales tasks as that person himself.
#     """
    
# system_message_prompt = SystemMessagePromptTemplate.from_template(template)
# human_template=f"{text}"
# human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
# chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
# print(chat_prompt)

response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"""You are a helpful marketing and sales assistant that asks me for your age, previous companies 
        you worked in and the projects that you lead. Then you are going to induce that personality into 
        you and you are going to help me with my marketing and sales tasks as that person himself.""",
        max_tokens=1000,
        temperature=0.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        n=1,
        stop=None
    )

    
res = response.choices[0].text.strip()
print(res)