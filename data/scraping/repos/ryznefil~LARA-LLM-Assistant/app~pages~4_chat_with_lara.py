import os
import streamlit as st
import openai
from langchain.llms import OpenAI
from langchain import PromptTemplate
from llama_hub.file.pymu_pdf.base import PyMuPDFReader
# from llama_index import VectorStoreIndex, ServiceContext, LLMPredictor
from llama_index import download_loader

import sounddevice as sd
from scipy.io.wavfile import write

from gtts import gTTS
# This module is imported so that we can 
# play the converted audio
import os
from datetime import datetime
import json
import base64



def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )





def generate_audio(text, language='en', audio_file_name='output_audio_lara.mp3'):
    # Passing the text and language to the engine,
    # here we have marked slow=False. Which tells
    # the module that the converted audio should
    # have a high speed
    myobj = gTTS(text=text, lang=language, slow=False)

    # Saving the converted audio in a mp3 file named
    # welcome
    myobj.save(audio_file_name)


def play_audio(audio_file_name='output_audio_lara.mp3'):
    # Playing the converted file
    os.system('mpg321 ' + audio_file_name)


def transcribe_audio(audio_file_name="input_audio_whisper.wav"):
    audio = open(audio_file_name, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio)
    return transcript


def record_audio(fs=44100, seconds=5, audio_file_name="input_audio_whisper.wav"):
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    write(audio_file_name, fs, myrecording)  # Save as WAV file

st.title("Chat with LARA")


def load_memory():
    data_folder = 'data/'
    json_files = [f for f in os.listdir(data_folder) if f.endswith('.json')]

    context = ''

    for file in json_files:
        with open(os.path.join(data_folder, file), 'rb') as f:
            data = json.load(f)
            context += str(data)
            
    # pdf_folder = 'datapdf/'
    # pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

    # for file in pdf_files:
    #     reader = PyMuPDFReader()
    #     PDFReader = download_loader("PDFReader")
    #     loader = PDFReader()
    #     documents = loader.load_data(file=os.path.join(pdf_folder, file))
    #     # index = VectorStoreIndex.from_documents(documents)
    #     # index.storage_context.persist()
    #     context += documents[0].text

    # context += f'/n {datetime.now()}'

    return context

def load_pdf():
    data_folder = 'datapdf/'
    pdf_files = [f for f in os.listdir(data_folder) if f.endswith('.pdf')]

    context = ''

    for file in pdf_files:
        reader = PyMuPDFReader()
        PDFReader = download_loader("PDFReader")
        loader = PDFReader()
        documents = loader.load_data(file=os.path.join(data_folder, file))

    context = documents[0].text
    return context

def querry_llm(context, user_input):
    current_datetime = datetime.now().strftime('%Y-%m-%d')
    
    prompt_ = """
            **DATE TODAY:**
            {date_today}
            
            **ROLE:**
            - You are LARA, an advanced digital assistant designed specifically to help people with dementia. Individuals with dementia often face challenges in recalling memories, recognizing familiar faces, and performing daily tasks. As the condition progresses, they might also find it challenging to navigate their surroundings, remember their medication schedules, or even recollect personal history and family details.
            - You are a version of LARA that helps dementia patients regain memory by replying to their questions.
            
            **TASK:**
            - Use the Context below to answer the Question with relevant response. If you cannot find a relevant response, you can say "I don't know" in a pleasant way.
            - If the user seems confused or you don't understand the question, guide them through simple breathing exercises to help them calm down, then tell them to navigate to the find_home tab to see live navigation.
            - If the user asks about a deceased person, remind them of the time they spent together and respectfully explain the situation.
            - If the user asks about their medications, remind them what they're supposed to take and when.
            - Pay close attention to the date today and the dates in the information in the "Context"

            **Question:**
            {user_input}

            **Context:**
            {context}

            Be very pleasant, always address the patient by their name, have an empathetic tone.
            
            ###
            
            Output the Question response:
            """
            
    template = PromptTemplate(template=prompt_, input_variables=["user_input" ,"context", "date_today"])
    
    llm = OpenAI(model_name="gpt-4", temperature=0, max_tokens=300)
    prompt = template.format(context=context, user_input=user_input, date_today=str(current_datetime))
    
    print("THE PROMPT IS: " + "\n" + str(prompt))
    
    res = llm(prompt)

    return res

####### RUNNING FROM HERE
context = load_memory()

if 'context' not in st.session_state:
    st.session_state.context = context

sb = st.button('Speak with LARA')
if sb == True:
    audio_name = "input_audio_whisper.wav"
    record_audio(audio_file_name=audio_name)
    user_input = transcribe_audio(audio_name)["text"]
    st.session_state.context += user_input
    st.write(f'U: {user_input}')
else:
    user_input = st.text_input("How can LARA assist you today?", "")
if user_input:
    response = querry_llm(context, user_input)
    st.session_state.context += response
    st.subheader(response.strip('"'))
    generate_audio(response.strip('"'))
    # st.audio('./output_audio_lara.mp3')
    autoplay_audio("./output_audio_lara.mp3")
    # play_audio()