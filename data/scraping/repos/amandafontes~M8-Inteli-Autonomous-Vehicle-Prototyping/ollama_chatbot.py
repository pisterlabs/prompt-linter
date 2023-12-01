import gradio as gr
from langchain.llms import Ollama

model = Ollama(base_url="http://localhost:11434", model="safety-specialist")

def load_model(historico, texto):
    print('Carregando o modelo...')
    resposta = model(texto)
    historico.append(texto, modelo)
    return "", historico

with gr.Blocks() as interface:
    title = "Chatbot especialista em segurança do trabalho"
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Digite aqui a sua dúvida sobre segurança do trabalho em ambientes industriais.")
    clear = gr.ClearButton([msg, chatbot])

    msg.submit(load_model, [chatbot, msg], [msg, chatbot])

if __name__ == "__main__":
    interface.launch()