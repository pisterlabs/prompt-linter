import streamlit as st
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

ANTHROPIC_API_KEY = st.secrets["apikey"]

anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)

st.set_page_config(page_title="Anthropic", page_icon="🧠")

prompt = st.chat_input("Say something")

pre_prompt = "You are an expert in NLP. Take a deep breath before you answer my question."

if prompt:
    st.write("You:", prompt)
    st.subheader("Anthropic response")
    response = anthropic.completions.create(
        model="claude-instant-1.2",
        max_tokens_to_sample=400,
        prompt=f"{HUMAN_PROMPT} {pre_prompt}  {prompt} {AI_PROMPT}",
    )
    out = response.completion
    st.write(out)
