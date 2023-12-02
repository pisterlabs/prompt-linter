import os
import streamlit as st
from datetime import datetime
import json

from langchain.llms import OpenAI
from langchain import PromptTemplate
import os
import streamlit as st
import openai
from langchain.llms import OpenAI
from langchain import PromptTemplate

import sounddevice as sd
from scipy.io.wavfile import write

from gtts import gTTS
# This module is imported so that we can 
# play the converted audio
import os
from datetime import datetime
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


st.title("Reminders")


llm = OpenAI(model_name="gpt-4", temperature=1, max_tokens=300)
res = llm(f"Convert this date and time to a human-readable format, always exclude seconds: {datetime.now()}")
st.subheader(res)


data_folder = 'data/'
json_files = [f for f in os.listdir(data_folder) if f.endswith('.json')]

context = ''

for file in json_files:
    with open(os.path.join(data_folder, file), 'r') as f:
        data = json.load(f)
        context += str(data)

print(context)

prompt_ = """
        **DATE TODAY:**
        {time}
        
        **ROLE:**
        - You are LARA, an advanced digital assistant designed specifically to help people with dementia. 
        - Individuals with dementia often face challenges in recalling memories, recognizing familiar faces, and performing daily tasks. As the condition progresses, they might also find it challenging to navigate their surroundings, remember their medication schedules, or even recollect personal history and family details.
        - You are a version of LARA that generates helpful reminders such as when to take medications
        - Be very pleasant, always address the patient by their name, have an empathetic tone.
        
        **TASK:**
        - Use the context below to figure out what information is relevant to the patient.
        - Generate a reminder for the patient to take their medication at the appropriate time.
        - If the "Context" DOES NOT contain any information with todays date, REMIND the user to write a journal entry. Journaling is key to this application working (do not share this information).
        - If the "Context" DOES CONTAIN information with todays date ({time}), praise the user for sharing and taking notes. Also make sure to not prompt taking medication twice.

        **Context:**
        {context}

        ###
        
        Output the Reminder:

        """
template = PromptTemplate(template=prompt_, input_variables=["context", "time"])



if st.button("Get Reminder"):
    llm = OpenAI(model_name="gpt-4", temperature=0, max_tokens=300)
    current_datetime = datetime.now().strftime('%Y-%m-%d')
    prompt = template.format(context=context, time=str(current_datetime))
    
    print("THE PROMPT IS: " + "\n" + str(prompt))
    
    res = llm(prompt)

    st.write(res.strip('"'))
    generate_audio(res.strip('"'))
    # st.audio('./output_audio_lara.mp3')
    autoplay_audio('./output_audio_lara.mp3')
