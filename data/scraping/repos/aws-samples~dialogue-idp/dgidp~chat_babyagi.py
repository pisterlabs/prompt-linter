#from transformers import pipeline
import time
import os
import openai
import magic
import requests
import boto3
import json
import pickle
import gradio as gr
from handler import openai_asr, langchain_idp, pdf_2_text


# start_sequence = "\nAI:"
# restart_sequence = "\nHuman: "
# last_message = prompt
prompt = "How can I help you today?"

block = gr.Blocks()
with block:
    gr.HTML(
        f"""
          <div class="main-div">
            <div>
               <header>
               <h2>Dialogue Guided Intelligent Document Processing</h2>
               </header>
               <p>Dialogue Guided Intelligent Document Processing (DGIDP) is an innovative approach to extracting and processing information from documents by leveraging natural language understanding and conversational AI. This technique allows users to interact with the IDP system using human-like conversations, asking questions, and receiving relevant information in real-time. The system is designed to understand context, process unstructured data, and respond to user queries effectively and efficiently.</p> <p>While the text or voice chat accepts all major languages, the document upload feature only accepts files in English, German, French, Spanish, Italian, and Portuguese. The demo supports <u>multilingual text and voice</u> input, as well as <u>multi-page</u> documents in PDF, PNG, JPG, or TIFF format.</p>
            </div>
            <a href="https://www.buymeacoffee.com/alfredcs" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" height="32px" width="108px" alt="Buy Me A Coffee"></a>
            <br>
          </div>
        """
    )
    model_choice = gr.Dropdown(choices=["dgidp", "gpt-3.5", "babyagi", "bloom", "j2-jumbo-instruct", "flan-t5-xl", "bedrock (coming soon)", "gpt4all (coming soon)", "gpt-4 (coming soon)"], label="Model selection", value="gpt-3.5")
    gr.HTML(f"""<hr style="color:blue>""")
    #file1 = gr.File(file_count="single")
    #upload = gr.Button("OCR")
    gr.HTML(f"""<hr style="color:blue>""")
    chatbot = gr.Chatbot().style(height=1750)
    #message = gr.Textbox(placeholder=prompt, lines=1)
    #audio = gr.Audio(source="microphone", type="filepath", show_label=True,height=550)
    #file1 = gr.File(file_count="single")
    state = gr.State()
    with gr.Row().style(equal_height=True):
      with gr.Column():
        message = gr.Textbox(placeholder=prompt, show_label=True)
        #textChat = gr.Button("Text Chat")
      with gr.Column():
        audio = gr.Audio(source="microphone", type="filepath", show_label=True)
        #voiceChat = gr.Button("Voice Chat")
    with gr.Row().style(equal_height=True):
      with gr.Column():
        textChat = gr.Button("Text Chat")
      with gr.Column():
        voiceChat = gr.Button("Voice Chat")
    with gr.Row().style(equal_height=True):
      with gr.Column():
        file1 = gr.File(file_count="single")
      with gr.Column():
        file1_img = gr.Image(type="filepath", label="Upload an Image")

    upload = gr.Button("Transcribe")
    state = gr.State()
    textChat.click(langchain_idp, inputs=[message, state, model_choice], outputs=[chatbot, state])
    voiceChat.click(openai_asr, inputs=[audio, state, model_choice], outputs=[chatbot, state])
    upload.click(pdf_2_text, inputs=[file1, state], outputs=[chatbot, state])
    #clear.click()

block.launch(ssl_keyfile=os.environ.get('KEY_PATH'), ssl_certfile=os.environ.get('CERT_PATH'), ssl_verify=False, debug=True, server_name="0.0.0.0", server_port=7862, height=2048, share=False)
