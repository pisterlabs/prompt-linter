import random
import gradio as gr
from langchain.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap, RunnablePassthrough

model = Ollama(model="mistral")
prompt = ChatPromptTemplate.from_template(
"""
You are now my personal travel agent. Act as someone who has immense travel
experience and knows the best places in the world to do certain activities. I
want to know where I should go to {activity}. Give the answers as a list of
items, no bigger than 5 items. For each item, create a simple sentence
justifying this choice.
"""
)

chain = {"activity": RunnablePassthrough()} | prompt | model


def response(message, history):
    print(message)
    msg = ""
    for s in chain.stream(message):
        print(s, end="", flush=True)
        msg += s
        yield msg

demo = gr.ChatInterface(response).queue()

demo.launch()



