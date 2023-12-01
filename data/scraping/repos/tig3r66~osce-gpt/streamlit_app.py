import streamlit as st
import time
import numpy as np
import json

# OpenAI stuff
import openai
from transformers import GPT2TokenizerFast
TOKENIZER = GPT2TokenizerFast.from_pretrained("gpt2")

# audio detection & transcription
import whisper
import speech_recognition as sr

# text to speech
import io
import gtts
import ffmpeg
from pydub import AudioSegment
from pydub.playback import play

# environment variables
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


with open('cases.json') as f:
    CASES = json.load(f)


class Patient:

    def __init__(self, instructions, option=None):
        self.option = option.strip().lower()
        self.memory = [{"role": "system", "content": instructions}]
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokens = len(self.tokenizer(instructions)['input_ids'])
        self.max_tokens = 4000
        self.r = sr.Recognizer()
        self.model = whisper.load_model("base")
        self.history = []
        self.images = CASES[self.option.strip().lower()][0]['images'] if len(CASES[self.option.strip().lower()][0]['images']) > 0 else None
        with sr.Microphone() as source:
            self.r.adjust_for_ambient_noise(source, duration=2)

    def load_audio(self, file, sr=16000):
        if isinstance(file, bytes):
            inp = file
            file = 'pipe:'
        else:
            inp = None
        try:
            out, _ = (
                ffmpeg.input(file, threads=0)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, input=inp)
            )
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
        return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    def transcribe(self, audio):
        try:
            return self.model.transcribe(self.load_audio(audio.get_wav_data()), language='en', fp16=False)['text']
        except:
            return None

    def generate_response(self, prompt):
        self.update_memory("user", prompt)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            # model='gpt-4',
            messages=self.memory,
            temperature=0.5,
            top_p=1,)['choices'][0]['message']['content']
        self.update_memory("assistant", response)
        return response

    def generate_response_stream(self, memory):
        response = openai.ChatCompletion.create(
            # model="gpt-3.5-turbo",
            model='gpt-4',
            messages=memory,
            temperature=0.5,
            top_p=1,
            stream=True)
        return response

    def speak(self, text):
        audio = io.BytesIO()
        gtts.gTTS(text=text).write_to_fp(audio)
        audio.seek(0)
        play(AudioSegment.from_file(audio, format="mp3"))

    def update_memory(self, role, content):
        self.memory.append({"role": role, "content": content})
        self.tokens += len(self.tokenizer(content)['input_ids'])
        while self.tokens > self.max_tokens:
            popped = self.memory.pop(0)
            self.tokens -= len(self.tokenizer(popped['content'])['input_ids'])

    def main(self):
        st.write("**Clinical scenario initialized. You may begin speaking now.** You can end the scenario by clicking the *stop* button.")
        stop_button = st.button('Stop', disabled=st.session_state.feedback_state, on_click=feedback)
        st.markdown("---")

        if st.session_state.feedback_state is False:
            with sr.Microphone() as source:
                source.pause_threshold = 1  # silence in seconds
                while True:
                    if stop_button:
                        break
                    audio = self.r.listen(source)
                    text = self.transcribe(audio)
                    if text:
                        st.write(f'Me: {text}')
                        response = self.generate_response(text)
                        self.history.append(f'Me: {text}')
                        self.history.append(f'Patient: {response}')
                        update_history(f'Me: {text}')
                        update_history(f'Patient: {response}')
                        st.write(f"Patient: {response}")
                        self.speak(response)
        else:
            for i in st.session_state.history:
                st.write(i)

        st.markdown("---")
        st.write('*Clinical scenario ended.* Thank you for practicing with OSCE-GPT! If you would like to practice again, please reload the page.')

        # feedback and SOAP note
        col1, col2 = st.columns(2)
        with col1:
            st.write('If you would like feedback, please click the button below.')
            feedback_button = st.button('Get feedback', key='feedback')
        with col2:
            st.write('If you would like to create a SOAP note from this conversation, please click the button below.')
            soap_button = st.button('Get SOAP note', key='soap_note')
        if feedback_button:
            if len(st.session_state.history) != 0:
                instructions = 'Based on the chat dialogue between me and the patient, please provide constructive feedback and criticism for me, NOT the patient. Comment on things that were done well, areas for improvement, and other remarks as necessary. For example, patient rapport, conversation organization, exploration of a patient\'s problem, involvement of the patient in care, explanation of reasoning, appropriate clinical reasoning, and other aspects of the interaction relevant to a patient interview. If relevant, suggest additional questions that I could have asked. Do not make anything up.'
                temp_mem = [{'role': 'user', 'content': '\n'.join(st.session_state.history) + instructions}]
                stream = self.generate_response_stream(temp_mem)
                t = st.empty()
                full_response = ''
                for word in stream:
                    try:
                        next_word = word['choices'][0]['delta']['content']
                        full_response += next_word
                        t.write(full_response)
                    except:
                        pass
                    time.sleep(0.001)
            else:
                st.write('No conversation to provide feedback on.')
        if soap_button:
            if len(st.session_state.history) != 0:
                instructions = 'Based on the chat dialogue between me and the patient, please write a SOAP note. Use bullet points for each item. Use medical abbreviations and jargon as appropriate (e.g., PO, BID, NPO). Do not make anything up.'
                temp_mem = [{'role': 'user', 'content': '\n'.join(st.session_state.history) + instructions}]
                stream = self.generate_response_stream(temp_mem)
                t = st.empty()
                full_response = ''
                for word in stream:
                    try:
                        next_word = word['choices'][0]['delta']['content']
                        full_response += next_word
                        t.write(full_response)
                    except:
                        pass
                    time.sleep(0.001)
            else:
                st.write('No conversation to provide feedback on.')


def disable():
    st.session_state.disabled = True

def feedback():
    st.session_state.feedback_state = True

def update_history(prompt):
    st.session_state.history.append(prompt)
    tokens = len(TOKENIZER('\n'.join(st.session_state.history))['input_ids'])
    max_tokens = 8000
    while tokens > max_tokens:
        st.session_state.history.pop(0)

def create_prompt(cases, option):
    instructions = f"Instructions: You are a patient talking to a physician. Use the provided context to answer the physician. Do not give too much away unless asked. You may use creativity in your answers.\n\nContext:{cases[option.strip().lower()][0]['case_info']}"

    images = cases[option.strip().lower()][0]['images']
    if len(images) > 0:
        instructions += "\nPlease write the image files in parentheses if you would like to use them in your questions. If you've already used an image, no need to include it in parentheses."
        image_prompt = '\n\nImages:\n'
        for i in range(len(images)):
            image_prompt += cases[option.strip().lower()][0]['images'][i]
            image_prompt += ': '
            image_prompt += cases[option.strip().lower()][0]['img_descriptions'][i]
            image_prompt += '\n'
    else:
        image_prompt = ''

    return instructions + image_prompt

def main():
    st.title('OSCE-GPT')
    st.caption('Powered by Whisper, GPT-4, and Google text-to-speech.')
    st.caption('By [Eddie Guo](https://tig3r66.github.io/)')

    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'disabled' not in st.session_state:
        st.session_state.disabled = False
    if 'feedback_state' not in st.session_state:
        st.session_state.feedback_state = False

    option = st.selectbox(
        "Which clinical scenario would you like to practice with?",
        ("Select one", *list(CASES.keys())),
        disabled=st.session_state.disabled,
        on_change=disable,
    )

    while option == "Select one":
        time.sleep(1)

    if option.lower() == "breaking bad news":
        st.write("You are seeing a 54 year old woman named Angela who has had headaches, seizures, and memory loss. The MRI scan showed a rapidly growing brain tumour. The pathology report of the biopsy showed the tumour is glioblastoma multiforme. Please deliver this news to the patient.")
        time.sleep(3)

    st.write(f'You selected: {option.lower()}')
    prompt = create_prompt(CASES, option)
    patient = Patient(prompt, option)
    patient.main()



if __name__ == '__main__':
    main()
