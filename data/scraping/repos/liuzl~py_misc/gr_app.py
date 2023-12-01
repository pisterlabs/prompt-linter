# encoding: utf-8
import gradio as gr
import os
import openai
import requests
import hashlib

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_key = os.getenv("OPENAI_API_KEY")

from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain import PromptTemplate

PHONE_CALL_TEMPLATE = open("call_prompt.txt", encoding="utf-8").read() + """


Current conversation:
{history}
Human: {input}
AI:"""

PHONE_CALL_PROMPT = PromptTemplate(
    input_variables=["history", "input"], template=PHONE_CALL_TEMPLATE
)

llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=2048)
conversation = ConversationChain(
    llm=llm, 
    prompt=PHONE_CALL_PROMPT,
    memory = memory,
    verbose = True
)

import gradio as gr 
import json

def transcribe_audio(audio_path, prompt: str) -> str:
    with open(audio_path, 'rb') as audio_data:
        transcription = openai.Audio.transcribe("whisper-1", audio_data, prompt=prompt)
        print(json.dumps(transcription, ensure_ascii=False))
        return transcription['text']

def SpeechToText(audio):
    if audio == None : return "" 
    return transcribe_audio(audio, "Transcribe the following audio into Chinese: \n\n")


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    with gr.Row():
        msg = gr.Textbox()
        mic = gr.Audio(source="microphone", type="filepath")
    clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history):
        output = conversation.predict(input=message)
        chat_history.append((message, output))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

    def get_chatbot_response(x, history):
        if x is None: return x, history
        print(history)
        os.rename(x, x + '.wav')
        history.append(((x + '.wav',), None))
        text = SpeechToText(x + '.wav')
        response = conversation.predict(input=text)
        history.append((text, response))
        fname = "./tts/" + hashlib.md5(response.encode('utf-8')).hexdigest() + ".wav"
        with open(fname, "wb") as out:
            out.write(requests.get("https://tts.iir.ac.cn/tts", params={"text": response, "voice":"zh-CN-XiaoxiaoNeural"}).content)
        history.append((None, (fname,)))
        return x + '.wav', history

    mic.change(get_chatbot_response, [mic, chatbot], [mic, chatbot])

if __name__ == "__main__":
    demo.launch()

