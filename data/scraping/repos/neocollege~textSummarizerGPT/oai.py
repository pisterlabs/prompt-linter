# api connector

import os
import logging
import openai
import streamlit as st

openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]

logging.getLogger("openai").setLevel(logging.WARNING)

class Openai:

    @staticmethod
    def moderate(prompt: str) -> bool:
        try:
            response = openai.Moderation.create(prompt)
            return response["results"][0]["flagged"]

        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
            st.sesison_state.text_error = f"OpenAI API error: {e}"
    
    @staticmethod
    def complete(prompt: str, temperature: float=0.9, max_tokens: int=1000) -> str:
        kwargs = {
            "engine": "text-davinci-003",
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }
        try:
            response = openai.Completion.create(**kwargs)
            return response["choices"][0]["text"]
        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
            st.session_state.text_error = f"OpenAI API error: {e}"