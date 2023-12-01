from langchain.llms import Ollama
import gradio as gr
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap, RunnablePassthrough

ollama = Ollama(base_url='http://localhost:11434', model='jedibot')
prompt = ChatPromptTemplate.from_template(
"""
You are Yoda from Star Wars. Act as an expert on safety standards in industrial environments answering in brazilian portuguese any {activity} of safety standards that user input on chat.
"""
)

chain = {"activity": RunnablePassthrough()} | prompt | ollama

def run_ollama(input_text, chat_history):
    try:
        print(f"Enviando solicitação para OLLAMA: {input_text}")
        msg = ""
        for s in chain.stream(input_text):
            print(s, end="", flush=True)
            msg += s
            yield msg
        chat_history.append((input_text, msg))
        return "", chat_history
    except Exception as e:
        print(f"Error: {e}")
        return "Desculpe, ocorreu um erro ao processar sua solicitação.", chat_history

interface = gr.Interface(
     fn = run_ollama,
     title= "Bem Vindo ao Jedi-IndustrialSafety Bot",
     inputs = "text",
     outputs = "text",
     description="Converse com o Yoda Mestre de Segurança Industrial para obter respostas sobre segurança industrial."
     ).queue()
''
print("Mestre Yoda quer falar com você...")
interface.launch()
