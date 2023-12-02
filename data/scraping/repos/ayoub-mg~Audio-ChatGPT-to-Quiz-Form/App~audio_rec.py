import streamlit as st
import sounddevice as sd
import soundfile as sf
import requests
import json
import whisper
import openai
import re

from apiclient import discovery
from httplib2 import Http
from oauth2client import client, file, tools


# Define your OpenAI API key and endpoint
API_KEY = "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"
API_ENDPOINT = "https://api.openai.com/v1/engines/whisper/betas/0.3.0/completions"

def record_audio(filename, duration):
    # Set the audio parameters
    sample_rate = 16000  # Sample rate in Hz
    channels = 1  # Mono audio

    # Record audio
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels)
    sd.wait()

    # Save the recorded audio to a file
    sf.write(filename, recording, sample_rate)

    st.success(f"Audio saved as {filename}.")

def transcribe_audio(filename):
    model = whisper.load_model("base")
    result = model.transcribe(filename)
    return result["text"]

    

def main():
    st.title("Audio ChatGPT To Quiz Form")

    duration = st.slider("Recording Duration (seconds)", min_value=1, max_value=10, value=5)
    n = st.slider("Number of Quiz Questions ", min_value=1, max_value=20, value=10)
    filename = st.text_input("Enter the filename to save the audio", "audio.wav")


    if st.button("Record"):
        record_audio(filename, duration)
        st.info("Recording complete.")

    
    if st.button("Transcribe"):
        st.info("Transcribing audio...")
        global transcription
        transcription = transcribe_audio(filename)
        st.success("Transcription:")
        st.write(transcription)   
        st.info("Generating form...")
        Link = form_generator(transcription,n)
        st.success("Link to the form:")
        st.write(Link)



def form_generator(text,n):
    openai.api_key = API_KEY
    content = text + f""" The quiz should contain {n} questions with multiple choice answers in the same format as the following, and in the end write finished : 
        Title of quiz : "quiz on ..." 
    Question 1 : ....
        a.  
        b.  
        c.
        d.
    Question 2 : ....
        a. 
        b. 
        c. 
        d. 
    """

    completion = openai.ChatCompletion.create(model = "gpt-3.5-turbo", messages = [{"role": "user", "content": content }])
    requests = completion.choices[0].message.content

    # Parsing the result

    Title = re.findall(r'Quiz on (.*?)\n' , requests)
    questions = re.findall(r'Question \d+: (.*?)\n', requests)

    a_answers = re.findall(r'a\.(.*?)\n' ,requests)
    b_answers = re.findall(r'b\.(.*?)\n' ,requests)
    c_answers = re.findall(r'c\.(.*?)\n' ,requests)
    d_answers = re.findall(r'd\.(.*?)\n' ,requests)


    q = []

    for i in range(n):
        q.append({
            "createItem": {
                "item": {
                    "title": questions[i],
                    "questionItem": {
                        "question": {
                            "required": True,
                            "choiceQuestion": {
                                "type": "RADIO",
                                "options": [
                                    {"value": a_answers[i]},
                                    {"value": b_answers[i]},
                                    {"value": c_answers[i]},
                                    {"value": d_answers[i]}
                                ],
                                "shuffle": True
                            }
                        }
                    },
                },
                "location": {
                    "index": i
                }
            }
        })


    SCOPES = "https://www.googleapis.com/auth/forms.body"
    DISCOVERY_DOC = "https://forms.googleapis.com/$discovery/rest?version=v1"

    store = file.Storage('token.json')
    creds = None
    if not creds or creds.invalid:
        flow = client.flow_from_clientsecrets('key.json', SCOPES)
        creds = tools.run_flow(flow, store)

    form_service = discovery.build('forms', 'v1', http=creds.authorize(
        Http()), discoveryServiceUrl=DISCOVERY_DOC, static_discovery=False)

    # Request body for creating a form
    NEW_FORM = {
        "info": {
            "title": Title[0],
        }
    }

    # Request body to add a multiple-choice question
    NEW_QUESTION = {
        "requests": q}

    # Creates the initial form
    result = form_service.forms().create(body=NEW_FORM).execute()

    # Adds the question to the form
    question_setting = form_service.forms().batchUpdate(formId=result["formId"], body=NEW_QUESTION).execute()

    # Prints the result to show the question has been added
    get_result = form_service.forms().get(formId=result["formId"]).execute()
    return get_result["responderUri"]
    


if __name__ == "__main__": 
    main()

