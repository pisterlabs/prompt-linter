import asyncio
import re
from playwright.async_api import async_playwright
from playwright.sync_api import Page, expect

import Constants
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from langchain.prompts.chat import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI


#Yash's API key
service_key = Constants.SERVICE_KEY
llm = OpenAI(openai_api_key=service_key)

#PromptTemplate is better than simple string 
# prompt = PromptTemplate.from_template("From this raw HTML tag for a pizza store website, what is the most relevant tag I can use to order a pizza: {html}?")
# prompt.format(html="<h1>hello world</h1>?")

template = """
You are a helpful assistant with the objective of ordering a pizza.
From the following HTML tags which one should I click to start ordering a pizza.
"""
human_template = "Here is the html: {links}"
    
async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        base_url = "https://pizzapizza.ca/en"
        await page.goto(base_url)
        
        await browser.close()

asyncio.run(main())

# import Constants
# from langchain.llms import OpenAI
# from langchain.chat_models import ChatOpenAI
# from langchain.prompts import PromptTemplate

# #Yash's API key
# service_key = Constants.SERVICE_KEY
# llm = OpenAI(openai_api_key=service_key)

# #PromptTemplate is better than simple string
# #Allows us to pass in different HTML strings with the 'same' prompt
# prompt = PromptTemplate.from_template("From this raw HTML tag for a pizza store website, what is the most relevant tag I can use to order a pizza: {html}?")
# prompt.format(html="<h1>hello world</h1>?")

# response = llm.predict(prompt)
# print(response)