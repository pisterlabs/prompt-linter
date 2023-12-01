from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import streamlit as st

ANTHROPIC_API_KEY = "API KEY HERE"

anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)

st.set_page_config(page_title="My GPT - by Shravan", page_icon="🤖")

st.title("AI Text Gen 🤖")
st.subheader("Write the Prompt")

prompt = st.text_area("Prompt", "Enter the Prompt Here")

check = st.button("Ask the AI")
if check:
    completion = anthropic.completions.create(
        model="claude-1",
        max_tokens_to_sample=500,
        prompt=f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}",
    )
    out = completion.completion
    st.write(out)
