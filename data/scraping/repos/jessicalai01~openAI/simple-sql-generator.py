import openai
import gradio as gr
from config import API_KEY
from sqlGenerator.contexts import contexts

AI_API_KEY = API_KEY
openai.api_key = AI_API_KEY
messages = contexts

def chat_function(chat_input):
    if chat_input:
        messages.append({"role":"user", "content":chat_input})
        chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages,temperature=1)
        reply = chat["choices"][0]["message"]["content"]
        messages.append({"role":"assistant","content":reply})
        return reply


inputs = gr.Textbox(lines=10, label="Intelligent SQL Generator: alisa")
outputs = gr.Textbox(label="Reply")

gr.Interface(fn=chat_function,
             inputs=inputs,
             outputs=outputs,
             title="Alisa is here for you").launch(share = True)
