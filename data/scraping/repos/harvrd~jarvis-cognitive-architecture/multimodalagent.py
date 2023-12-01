import os
import requests

from langchain.agents import initialize_agent, Tool
from langchain.tools.bing_search.tool import BingSearchRun, BingSearchAPIWrapper
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import PALChain, LLMChain
from langchain.llms import AzureOpenAI
# from langchain.utilities import ImunAPIWrapper, ImunMultiAPIWrapper
from langchain.utilities import SerpAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

MAX_TOKENS = 512

# llm = AzureOpenAI(deployment_name="text-chat-davinci-002", model_name="text-chat-davinci-002", temperature=0, max_tokens=MAX_TOKENS)
llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
memory = ConversationBufferMemory(memory_key="chat_history")
search = SerpAPIWrapper()

tools = [

    # Tool(
    #     name = "Layout Understanding",
    #     func=imun_layout.run,
    #     description=(
    #     "A wrapper around layout and table understanding. "
    #     "Useful after Image Understanding tool has recognized businesscard in the image tags."
    #     "This tool can find the actual business card text, name, address, email, website on the card."
    #     "Input should be an image url, or path to an image file (e.g. .jpg, .png)."
    #     )
    # ),
    #     Tool(
    #     name = "OCR Understanding",
    #     func=imun_read.run,
    #     description=(
    #     "A wrapper around OCR Understanding (Optical Character Recognition). "
    #     "Useful after Image Understanding tool has found text or handwriting is present in the image tags."
    #     "This tool can find the actual text, written name, or product name in the image."
    #     "Input should be an image url, or path to an image file (e.g. .jpg, .png)."
    #     )
    # ),        
    Tool(
        name = "search",
        func=search.run,
        description="Useful when you want to answer questions about current events or things found online"
    )
]

prompt = ChatPromptTemplate

chain = LLMChain(llm=llm, tools=tools, prompt="You are a chatbot")
# chain = initialize_agent(tools, llm, agent="conversational-assistant", verbose=True, memory=memory, return_intermediate_steps=True, max_iterations=4)
output = chain.conversation("https://www.oracle-dba-online.com/sql/weekly_sales_table.png")
