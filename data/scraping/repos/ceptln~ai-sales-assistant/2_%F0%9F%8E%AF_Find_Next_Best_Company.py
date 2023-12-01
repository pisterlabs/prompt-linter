import streamlit as st
from streamlit.components.v1 import html
import openai
import utils

# OpenAI setup
openai.api_key = utils.open_file('openai_api_key.txt')

# Set page config
utils.set_page_configuration()
utils.set_sidebar()

title = 'Find your next client'
subtitle = "Uncover lucrative prospects based on your existing clients"
mission = "Looking for your next deal to close? Just provide the name of some of your best clients and the AI will find some similiar companies."
utils.display_intro(title=title, subtitle=subtitle, mission=mission)

