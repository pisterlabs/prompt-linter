import os
import openai
import gradio as gr
from pyChatGPT import ChatGPT

session_token = "Token"



start_sequence = "\nAI:"
restart_sequence = "\nHuman: "

prompt = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today?\nHuman: "

def openai_create(prompt):

    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    temperature=0.9,
    max_tokens=150,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0.6,
    stop=[" Human:", " AI:"]
    )

    return response.choices[0].text



def chatgpt_clone(input, history):
    history = history or []
    s = list(sum(history, ()))
    s.append(input)
    inp = ' '.join(s)
    output = openai_create(inp)
    history.append((input, output))
    return history, history

def outputsGPT(input, history):
    resp = api1.send_message(input)
    history = history or []
    s = list(sum(history, ()))
    s.append(input)
    output = resp['message']
    history.append((input, output))
    print("Human: ", input)
    print("ChatGPT: ", output)
    return history, history


block = gr.Blocks()
api1 = ChatGPT(session_token) 

with block:
    try:
        chatbot = gr.Chatbot()
        message = gr.Textbox(placeholder=prompt)
        state = gr.State()
        submit = gr.Button("SEND")
        submit.click(outputsGPT, inputs=[message, state], outputs=[chatbot, state])
    except Exception as e:
        print("Error: ", e.message, e.args)

block.launch(debug = False, show_api=False)
