import os
import time

import openai
import arxiv
import google_serp_api
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import load_tools,initialize_agent,AgentType
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from spotipy import SpotifyClientCredentials

import gradio

load_dotenv()

openai.api_key= os.getenv('OPENAI_API_KEY')
google_serp_api.client = os.getenv('SERPAPI_API_KEY')
google_api_key = os.getenv('GOOGLE_CSE_ID')
google_cse_id = os.getenv('CUSTUM_SEARCH_ID')





lili = OpenAI(temperature=0.9)

tools = load_tools(["arxiv"],)
recherche = load_tools(["serpapi","llm-math"],llm=lili)


# search = GoogleSearchAPIWrapper()
#
# def top5_results(query):
#     return search.results(query, 5)
#
# tool = Tool(
#     name = "Google Search Snippets",
#     description="Search Google for recent results.",
#     func=top5_results
# )
#
#
# resultat = top5_results('One Piece 1083')
#
# for i in resultat:
#     print(i)

def reponse(prompt):
    agent_chain = initialize_agent(tools, lili, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, )
    bot_message = agent_chain
    time.sleep(2)
    return "", bot_message.run(prompt)

with gradio.Blocks() as app :
    chatbot = gradio.Chatbot()
    message = gradio.Textbox()
    clear_button = gradio.ClearButton([message,chatbot])

    message.submit(reponse,[message,chatbot],[message,chatbot])

app.launch()


#
# while True:
#     agent_chain.run(input('YMC:'))

#     tool.run(input("YMC: "))


