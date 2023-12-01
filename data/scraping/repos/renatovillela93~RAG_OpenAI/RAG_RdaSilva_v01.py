#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# RAG ChatGPT test based on amrss code "chatgpt-api-python" (GitHub)
# Reference: https://github.com/amrrs/chatgpt-api-python

# Import libraries
import openai
import gradio as gr

# API key
openai.api_key = "Your Key here"

messages = [
    {"role": "system", "content": "You are a helpful and kind AI Assistant."},
]

# Creating chatbot function
def chatbot(input):
    if input:
        messages.append({"role": "user", "content": input})
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
        reply = chat.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
        return reply

inputs = gr.inputs.Textbox(lines=7, label="Chat with AI")
outputs = gr.outputs.Textbox(label="Reply")

gr.Interface(fn=chatbot, inputs=inputs, outputs=outputs, title="AI Chatbot",
             description="Ask anything you want",
             theme="compact").launch(share=True)

