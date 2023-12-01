import io
import openai
import streamlit as st
from textblob import TextBlob
st.set_page_config(
    page_title="TANA"
)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.sidebar.title("Menu")

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://media.giphy.com/media/1BfRG8cK5SPOer97aK/giphy.gif");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 

openai.api_key = 'openai_api_key'
st.markdown("<h1 style='text-align: center; color: white;'>VoIP Call Recordings to Text</h1>", unsafe_allow_html=True)
audio_file = st.file_uploader("Upload an audio file", type=["mp3"])
# Check if an audio file has been uploaded
if audio_file:
    # Create a file-like object from the bytes object
    audio_file_object = io.BytesIO(audio_file.read())
    # Set the name attribute of the file object
    audio_file_object.name = audio_file.name
    # Call OpenAI API to translate audio to text
    transcript = openai.Audio.translate("whisper-1", audio_file_object)
    # Display the transcript
    st.write("Transcript:")
    st.write(transcript)
    #Sentiment analysis
    st.write("Sentiment polarity is: ")
    st.write(TextBlob(transcript['text']).sentiment.polarity)
    st.write("Sentiment subjectivity is: ")
    st.write(TextBlob(transcript['text']).sentiment.subjectivity)
    if((TextBlob(transcript['text']).sentiment.polarity)<0):
        st.write("The recorrded audio is negative and can be malicious")
    else:
        st.write("The text is okay")
