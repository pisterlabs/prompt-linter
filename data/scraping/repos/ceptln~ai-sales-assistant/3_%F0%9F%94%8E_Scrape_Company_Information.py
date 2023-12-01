import streamlit as st
from streamlit.components.v1 import html
import openai
import utils

# OpenAI setup
openai.api_key = utils.open_file('openai_api_key.txt')

# Set page config
utils.set_page_configuration()
utils.set_sidebar()

title = "Scrape Company Information"
subtitle = "Blabla"
mission = "Page 2 jeijrijrijrcjjijcirjicjricjirjcirjc ijir jci rjcirjcijr ijri rj ij irjirjc icr 🚀"
utils.display_intro(title=title, subtitle=subtitle, mission=mission)