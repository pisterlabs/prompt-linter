from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import openai

with st.form("logo"):
    prompt = st.text_area("What would you like to generate?")
    submitted = st.form_submit_button()
    if submitted:
        response = openai.Image.create(prompt=prompt, n=1, size="512x512")
        image_url = response["data"][0]["url"]
        st.markdown(f"""
            ![]({image_url})
            [Image]({image_url})
        """)
