from langchain.llms import Ollama
from langchain.schema import AIMessage, HumanMessage
from dotenv import load_dotenv
import gradio as gr

load_dotenv()

ollama = Ollama(base_url='http://localhost:11434', model="dexter")

history = ""

def predict(message, history):
    gpt_response = ollama(message)
    return gpt_response

gr.ChatInterface(predict).launch()
