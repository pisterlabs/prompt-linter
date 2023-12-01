import base64
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
import openai
import tempfile
from pydub import AudioSegment
import io
import os

# Constants
ICONS = {
    "Voice Input": "./icons/microphone_icon.png",
    "Text Input": "./icons/keyboard_icon.png",
    "Sign Language Input": "./icons/hand_icon.png"
}

# Define function to create a data URI for an image
def image_to_data_uri(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Load icons
mic_icon = image_to_data_uri("./icons/microphone_icon.png")
keyboard_icon = image_to_data_uri("./icons/keyboard_icon.png")
hand_icon = image_to_data_uri("./icons/hand_icon.png")

def display_input_method_page():
    left_co,cent_co,last_co = st.columns(3)
    with cent_co:
        st.image("./icons/petlogo.jpg", width=200)
    st.markdown(f'<h1 style="text-align:center;color:#f46a4c;font-size:42px;">{"Choose Your Input Method"}</h1><br>', unsafe_allow_html=True)
    
    
    for method, icon_path in ICONS.items():
        icon_data = image_to_data_uri(icon_path)
        
        col1, col2 = st.columns([1, 3])
        if col2.button(method, key=method):
            st.session_state['input_method'] = method.lower().replace(' ', '_')
        col1.image(icon_path, width=50) 
        # Custom CSS for Streamlit buttons can be added via st.markdown, similarly to previous examples


def handle_input_method(input_method):
    if input_method == 'voice_input':
        # Handle voice input
        handle_voice_input()
    elif input_method == 'text_input':
        # Handle text input
        handle_text_input()
    elif input_method == 'sign_language_input':
        # Handle sign language input
        handle_sign_language_input()
    else:
        st.error("Invalid input method selected.")

# Set child-friendly styles
st.markdown(
    """
    <style>
        body {
            background-color: #FFDAB9; /* Peach background */
            font-family: 'Comic Sans MS', 'Chalkboard SE', 'Marker Felt'; /* Fun font */
        }
        h1, h2 {
            color: #E69966; /* Bright  titles */
            text-align: center;
        }
        .icon-button {
            padding: 15px 30px;
            font-size: 20px;
            border: none;
            background-color: #FEC773; /*  background colour for buttons */
            cursor: pointer;
            display: block; /* Make buttons block-level elements */
            width: 80%; /* Set a fixed width */
            margin: 10px auto; /* Center the buttons and add margin */
            text-align: center;
            border-radius: 15px;
            transition: background-color 0.3s;
        }
        .icon-button img {
            width: 30px;
            height: 30px;
        }
        .icon-button:hover {
            background-color: #E69966; /* Chartreuse color on hover */
        }
        .stButton>button {
            font-size: 1.5em;
            padding: 0.5em 2em;
            margin: 1em;
            align:center;
            ...
        }
    </style>
    """,
    unsafe_allow_html=True,
)

def get_image_download_link(img_bytes, filename, text):
    b64 = base64.b64encode(img_bytes).decode()
    return f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'

# Function to generate image from text using OpenAI
def generate_image_from_text(user_input):
    if 'generated_image' not in st.session_state:
        try:
            with st.spinner('Generating image...'):
                openai.api_key = "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"
                res = openai.Image.create(
                    prompt=user_input,
                    n=1,
                    size="1024x1024",
                    response_format="b64_json"
                )
                generated_image_data = res['data'][0]['b64_json']
                st.session_state['generated_image'] = base64.b64decode(generated_image_data)
        except Exception as e:
            st.error("Oops! Something went wrong while generating the image. Please try again later.")
    # Display the image
    st.image(st.session_state['generated_image'], width=50, caption="Generated Image", use_column_width=True)

def convert_to_wav(audio_bytes):
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    temp_audio_filename = tempfile.mktemp(suffix=".wav")
    audio.export(temp_audio_filename, format="wav")
    return temp_audio_filename

def get_text_from_audio(audio_bytes):
    r = sr.Recognizer()
    # Convert audio to WAV format
    wav_filename = convert_to_wav(audio_bytes)    
    try:
        with sr.AudioFile(wav_filename) as source:
            audio_data = r.record(source)
            try:
                text = r.recognize_google(audio_data)
                return text
            except sr.UnknownValueError:
                st.error("Google Speech Recognition could not understand the audio.")
            except sr.RequestError:
                st.error("Could not request results from Google Speech Recognition service.")
    finally:
        # Ensure the temporary file is deleted
        os.remove(wav_filename)    
    return None

def generate_story(user_input):
    with st.spinner('Generating story...'):
        response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a story teller for children aged 3 to 7 years. \
        Use words that are easy to understand by your audience.\
        Your stories should spark the audience's imagination and curiosity \
        Avoid sad stories or mentioning death. Try integrating the body senses like touch, hearing or smell into the stories. \
        Use Scottish English and names."},
            {"role": "user", "content": f"Given the prompt {user_input} write me a short and funny story, no longer than 1000 characters long"}
        ]
        )
        output = response['choices'][0]['message']['content']
    st.write(output)
    

def handle_voice_input():
    audio_bytes = audio_recorder()
    if audio_bytes:
        user_input = get_text_from_audio(audio_bytes)
        if user_input:
            generate_image_from_text(user_input)
            if st.button("Generate Story"):
                generate_story(user_input)

def handle_text_input():
    left_co,cent_co,last_co = st.columns(3)
    with cent_co:
        st.image("./icons/petlogo.jpg", width=200)
    st.markdown(f'<h1 style="color:#f46a4c;font-size:36px;">{"Type here:"}</h1>', unsafe_allow_html=True)
    user_input = st.text_input("")

    if user_input:
        st.session_state['user_input'] = user_input
        generate_image_from_text(user_input)
        if st.button("Generate Story"):
            generate_story(user_input)
        st.button("Play with Pet!")
    

def handle_sign_language_input():
    st.camera_input("Capture Makaton Sign Language")


def main():
   if 'input_method' not in st.session_state:
       display_input_method_page()
   else:
       handle_input_method(st.session_state['input_method'])
       


if __name__ == "__main__":
    main()