import random
import gradio as gr
from langchain.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap, RunnablePassthrough

model = Ollama(model="inspetor_bugiganga")
prompt = ChatPromptTemplate.from_template(
"""
Retrieve information on safety standards in industrial environments. Provide details on regulations, guidelines, and best practices to ensure a secure working environment in industrial settings. Include information on any recent updates or changes in safety protocols. Summarize key points and emphasize the importance of compliance with these standards for the well-being of workers and the overall safety of industrial operations.

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