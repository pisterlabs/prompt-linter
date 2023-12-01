from openai import OpenAI
import os, sys
import streamlit as st
from . import prompts

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

class Tutor():
    def __init__(self):
        p = prompts.Prompts()
        self.subject = p.SUBJECT()
        self.history = [{"role": "system", "content": p.INIT_PROMPT()}, \
                        {"role": "user", "content": p.FIRST_PROMPT()}]

    def ask(self, question): # method is a generator method to allow streaming OpenAI responses
        question = {"role": "user", "content": question}
        self.history.append(question)
        response = client.chat.completions.create(model="gpt-4", messages=self.history, stream=True)
        collect_msgs = ""
        for chunk in response:
            msg = chunk.choices[0].delta.content or ""
            collect_msgs += msg
            yield msg
        
        self.history.append({"role": "assistant", "content": collect_msgs})

