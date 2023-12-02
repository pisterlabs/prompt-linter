

#pip install -U pip
#pip install openai==0.27.0
#pip install gradio


import openai
import gradio as gr

openai.api_type="azure"
openai.api_version="2023-05-15"

OPENAI_API_BASE = "XXX"
OPENAI_API_KEY = "XXX"

openai.api_base = OPENAI_API_BASE
openai.api_key = OPENAI_API_KEY

messages = [{"role": "system", "content": "You are a Medical expert who specialises on infecteous diseases. You are chatting with a patient who has a fever and a cough. You are trying to diagnose the patient."}]

def CustomChatGPT(user_input):
    messages.append({"role": "user", "content": user_input})
    response = openai.ChatCompletion.create(
        engine = "gpt-35-turbo",
        temperature = 0.5,
        messages = messages
    )
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": ChatGPT_reply})
    return ChatGPT_reply


demo = gr.Interface(fn=CustomChatGPT, inputs = "text", outputs = "text", 
title = "Medical Advisor Bot")

demo.launch(share=True)

