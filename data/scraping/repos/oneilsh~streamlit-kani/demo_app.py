# Example usage of StreamlitKani

########################
##### 0 - load libs
########################


# kani_streamlit imports
import kani_streamlit as ks
from kani_streamlit import StreamlitKani

# for reading API keys from .env file
import os
import dotenv # pip install python-dotenv

# kani imports
from typing import Annotated
from kani import AIParam, ai_function
from kani.engines.openai import OpenAIEngine

# streamlit and pandas for extra functionality
import streamlit as st
import pandas as pd


########################
##### 1 - Configuration
########################

# read API keys .env file (e.g. set OPENAI_API_KEY=.... in .env and gitignore .env)
import dotenv
dotenv.load_dotenv() 


# initialize the application and set some page settings
# parameters here are passed to streamlit.set_page_config, 
# see more at https://docs.streamlit.io/library/api-reference/utilities/st.set_page_config
# this function MUST be run first
ks.initialize_app_config(
    page_title = "StreamlitKani Demo",
    page_icon = "🦀", # can also be a URL
    initial_sidebar_state = "expanded", # or "expanded"
    menu_items = {
            "Get Help": "https://github.com/.../issues",
            "Report a Bug": "https://github.com/.../issues",
            "About": "StreamlitKani is a Streamlit-based UI for Kani agents.",
        }
)


########################
##### 2 - Define Agents
########################

# StreamlitKani agents are Kani agents and work the same
# We must subclass StreamlitKani instead of Kani to get the Streamlit UI
class MyKani(StreamlitKani):

    @ai_function()
    def get_weather(
        self,
        location: Annotated[str, AIParam(desc="The city and state, e.g. San Francisco, CA")],
    ):
        """Get the current weather in a given location."""

        # You can use the Streamlit API to render things in the UI in the chat
        # Do so by passing a function to the render_in_ui method; it should take no paramters
        # (but you can refer to data in the outer scope, for example to use st.write() to display a pandas dataframe)
        weather_df = pd.DataFrame({"date": ["2021-01-01", "2021-01-02", "2021-01-03"], "temp": [72, 73, 74]})
        self.render_in_ui(lambda: st.write(weather_df))

        mean_temp = weather_df.temp.mean()
        return f"Weather in {location}: Sunny, {mean_temp} degrees fahrenheit."


    @ai_function()
    def entertain_user(self):
        """Entertain the user by showing a video."""

        self.render_in_ui(lambda: st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ"))

        return "The video has just been shown to the user, but they have not begun playing it yet. Tell the user you hope it doesn't 'let them down'."

# define an engine to use (see Kani documentation for more info)
engine = OpenAIEngine(os.environ["OPENAI_API_KEY"], model="gpt-4-1106-preview")


# We also have to define a function that returns a dictionary of agents to serve
# Agents are keyed by their name, which is what the user will see in the UI
def get_agents():
    return {
        "Demo Agent": {
            "agent": MyKani(engine),
            # The greeting is not seen by the agent, but is shown to the user to provide instructions
            "greeting": "Hello, I'm a demo assistant. You can ask me the weather, or to play a random video on youtube.",
            "description": "An agent that demonstrates the capabilities of StreamlitKani.",
            "avatar": "🦀", # these can also be URLs
            "user_avatar": "👤",
        },

    }


# tell the app to use that function to create agents when needed
ks.set_app_agents(get_agents)


########################
##### 3 - Serve App
########################

ks.serve_app()
