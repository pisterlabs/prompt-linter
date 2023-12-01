import mermaid
from pathlib import Path


# New imports
import openpyxl
import streamlit as st

from langchain import SQLDatabase
from langchain.agents import AgentType
from langchain.agents import initialize_agent, Tool
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains import LLMMathChain, SQLDatabaseChain
from langchain.llms import OpenAI
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.agents import load_tools
from streamlit_agent.callbacks.capturing_callback_handler import playback_callbacks
from streamlit_agent.clear_results import with_clear_container
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.memory import ConversationBufferMemory

from product_search import search_for_products

import pandas as pd
import numpy as np
import pydeck as pdk
from mapbox import Geocoder, Directions
from datetime import datetime, timedelta
import geopy.distance
import os

DB_PATH = (Path(__file__).parent / "Chinook.db").absolute()


SAVED_SESSIONS = {
    "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?": "leo.pickle",
    "What is the full name of the artist who recently released an album called "
    "'The Storm Before the Calm' and are they in the FooBar database? If so, what albums of theirs "
    "are in the FooBar database?": "alanis.pickle",
}

def calculate_eta(start_location, end_location, shipping_method):
    shipping_time = shipping_times.get(shipping_method)
    if shipping_time is None:
        raise ValueError(f"Unknown shipping method: {shipping_method}")

    response = directions.directions([start_location[::-1], end_location[::-1]], 'mapbox/driving')
    travel_time = response.json()['routes'][0]['duration']  # in seconds
    order_time = datetime.now()
    eta = order_time + shipping_time + timedelta(seconds=travel_time)

    return eta.strftime('%Y-%m-%d %H:%M:%S')

# st.title("🦜 LangChain: Chat with search")
st.set_page_config(
    page_title="Lumos", page_icon="🦜", layout="wide", initial_sidebar_state="collapsed"
)

"# 🦜🔗 Lumos: Revolutionize Your Business"
openai_api_key = st.write(
    os.environ["OPENAI_API_KEY"] == st.secrets["OPENAI_API_KEY"],
)

# Tools setup
llm = OpenAI(temperature=0, openai_api_key=openai_api_key, streaming=True)
llm_math_chain = LLMMathChain.from_llm(llm)
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
db_chain = SQLDatabaseChain.from_llm(llm, db)
search_google = GoogleSerperAPIWrapper()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

tools = [
    # Tool(
    #    name="Mermaid",
    #    func=mermaid.run,
    #    description="will display the provided mermaid diagram to the user. Input should be a mermaid diagram"
    # ),
    Tool(
        name="Search Google",
        func=search_google.run,
        description="Search and learn about topic"
    ),
    Tool(
        name="Search Products",
        func=search_google.run,
        description="Search and find product vendors, locations, etc"
    ),
    Tool(
        name="Search Prices",
        func=search_google.run,
        description="Search and summarize prices for a set of vendors"
    ),
    Tool(
        name="Compute Prices",
        func=search_google.run,
        description="Find and estimate average cost of selling product from the business"
    ),
]

# tool_order_prompt = """
# The tools should be used in this order for answering questions:
# 1. SearchG - Use this to search the internet for general information
# 2. Calculator - Use this for any math calculations 
# 3. FooBar DB - Use this to look up information in the FooBar database
# 4. Mermaid - Use this to generate mermaid diagrams
# You should follow this order whenever possible when answering the user's questions.
# """

# search = initialize_agent(tools, llm, tool_order_prompt, verbose=True)
mrkl = initialize_agent(tools, llm, 
                        agent_path="./search_agent.json", 
                        verbose=True, memory=memory)
# mrkl = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

json_agent = initialize_agent(tools, llm, 
                        agent_path="./json_agent.json", 
                        verbose=True)

# Existing imports and setup
tabs = st.tabs(["Product Search", "QA", "Map"])

with tabs[0]:
    with st.form(key="form"):
        user_input = st.text_input("What type of business do you run?")
        # if not user_input:
        #     user_input = prefilled
        submit_clicked = st.form_submit_button("Submit Inquiry")
        ## Add langchain preprompt?

#justin code drop
with tabs[1]:
    df = pd.read_csv('coffeeshop.csv')

    st.dataframe(
        df,
        column_config={
            "product_name": "Product Name",
            "supplier_name": "Supplier Name",
            #"address": "Address",
            "price": "Price",
            "quality": "Quality",
            "environmental_score": st.column_config.NumberColumn(
                "Environmental Score",
                help="Stars",
                format="%d ⭐",
            ),
            "ETA": st.column_config.ProgressColumn(
                "ETA",
                help="ETA",
                format="%f miles",
                min_value=0,
                max_value=3000
            ),
        },
        hide_index=True,
    )

output_container = st.empty()
answer = ''
if with_clear_container(submit_clicked):
    output_container = output_container.container()
    output_container.chat_message("user").write(user_input)

    answer_container = output_container.chat_message("assistant", avatar="🦜")
    st_callback = StreamlitCallbackHandler(answer_container)

    answer = mrkl.run(user_input, callbacks=[st_callback])
    
    # os.environ["answer"] = answer
    answer_container.write(answer)


# jr = json_agent(os.environ["answer"])
# st.write(f'Exporting: {jr}')
