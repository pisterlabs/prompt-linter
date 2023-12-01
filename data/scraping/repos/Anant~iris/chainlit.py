import chainlit as cl
import os
from langchain import OpenAI, LLMMathChain, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool, ZeroShotAgent, agent_types
from langchain.chat_models import ChatOpenAI
from utils.giphy import GiphyAPIWrapper
from utils.foursquare import FoursquareAPIWrapper
from utils.knowledge import KnowledgeWrapper

@cl.langchain_factory(use_async=False)
def load():
    # Main llm for Chat 
    llm1 = ChatOpenAI(temperature=0, streaming=True)
    # LLM for Math Chain 
    llm = OpenAI(temperature=0, streaming=True)

    # Initialize additional tools 
    search = SerpAPIWrapper()
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
    # Custom tools 
    giphy = GiphyAPIWrapper()
    foursquare = FoursquareAPIWrapper()
    knowledge = KnowledgeWrapper()

    tools = [  
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions",
        ),      
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math",
        ),
        Tool(
            name="GiphySearch",
            func=giphy.run,
            return_direct=True,
            description="useful for when you need to find a gif or picture, and for adding humor to your replies. Input should be a query, and output will be an html embed code which you MUST include in your Final Answer."
        ),
        Tool(
            name="FoursquareSearch",
            func=foursquare.run,
            description="useful for when you need to find information about a store, park, or other venue. Input should be a query, and output will be JSON data which you should summarize and give back relevant information."
        ),
        Tool(
            name="FoursquareNear",
            func=foursquare.near,
            description="useful for when you need to find information about a store, park, or other venue in a particular location. Input should be a pipe separated list of strings of length two, representing what you want to search for and where. For example, `coffee shops| \"chicago il\"` would be the input if you wanted to search for coffee shops in or near chicago, illinois and output will be JSON data which you should summarize and give back relevant information."
        ),
        Tool(
            name="FailedBankList",
            func=knowledge.banks,            
            description="useful for when you need to find information about failed banks. the input should either names of banks or the years to see if there are matches. "
        ),   
        Tool(
            name="AmontilladoText",
            func=knowledge.amontillado,
            return_direct=True,
            description="useful for when you need to answer literary questions about The Cask of Amontillado, a short story by Edgar Alan Poe. This uses a vector similarity search. You should ask targeted questions to find similar documents. Input should be a query, and output will be an answer which you should summarize and give back relevant information."
        )       
    ]
    return initialize_agent(
        tools, llm1, agent=agent_types.AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )



"""
We can add this to the tool chain to see it work in Chainlit


        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions",
        ),

        ,
        Tool(
            name="GiphySearch",
            func=giphy.run,
            return_direct=True,
            description="useful for when you need to find a gif or picture, and for adding humor to your replies. Input should be a query, and output will be an html embed code which you MUST include in your Final Answer."
        ),
        Tool(
            name="FoursquareSearch",
            func=foursquare.run,
            description="useful for when you need to find information about a store, park, or other venue. Input should be a query, and output will be JSON data which you should summarize and give back relevant information."
        ),
        Tool(
            name="FoursquareNear",
            func=foursquare.near,
            description="useful for when you need to find information about a store, park, or other venue in a particular location. Input should be a pipe separated list of strings of length two, representing what you want to search for and where. For example, `coffee shops| \"chicago il\"` would be the input if you wanted to search for coffee shops in or near chicago, illinois and output will be JSON data which you should summarize and give back relevant information."
        ),
        Tool(
            name="FailedBankList",
            func=knowledge.banks,            
            description="useful for when you need to find information about failed banks. the input should either names of banks or the years to see if there are matches. "
        ),   
        Tool(
            name="AmontilladoText",
            func=knowledge.amontillado,
            return_direct=True,
            description="useful for when you need to answer literary questions about The Cask of Amontillado, a short story by Edgar Alan Poe. This uses a vector similarity search. You should ask targeted questions to find similar documents. Input should be a query, and output will be an answer which you should summarize and give back relevant information."
        )   
"""

    