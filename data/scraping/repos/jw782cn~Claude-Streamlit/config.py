import promptlayer
import streamlit as st
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT


# openai
# I use promptlayer to log my requests
promptlayer.api_key = st.secrets.get("PROMPTLAYER_API_KEY", "")
openai = promptlayer.openai
openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
openai.api_key = openai_api_key

# anthropic
anthropic_api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
anthropic = Anthropic(api_key=anthropic_api_key)
