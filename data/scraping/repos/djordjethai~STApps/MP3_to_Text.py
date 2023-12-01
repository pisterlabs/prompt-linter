# This code converts audio file to txt file. There is 25Mb limit for audio files

import openai
import os
import streamlit as st
from mojafunkcja import st_style, positive_login

st.set_page_config(
    page_title="MP3 to Text",
    page_icon="👉",
    layout="wide"
)
st_style()


def main():
    # Read OpenAI API key from env
    openai.api_key = os.environ.get('OPENAI_API_KEY')

    # App start
    st.subheader("Konvertujte MP3 🔊 u TXT 📄")
    audio_file = st.file_uploader("Choose a file")
    # transcript_json= "transcript"
    transcritpt_text = "transcript"
    if audio_file is not None:

        placeholder = st.empty()
        st.session_state['question'] = ''

        with placeholder.form(key='my_form', clear_on_submit=False):
            jezik = st.radio(
                "Odaberite jezik izvornog teksta ⤵️",
                key="jezik",
                options=["sr", "en", "th", "de", "fr", "hu",
                         "it", "ja", "ko", "pt", "ru", "es", "zh"],
                horizontal=True,
            )
            submit_button = st.form_submit_button(label='Submit')

            if submit_button:
                with st.spinner("Sačekajte trenutak..."):
                    transcript = openai.Audio.transcribe(
                        "whisper-1", audio_file, language=jezik)
                    # transcript_dict = {"text": transcript.text}
                    transcritpt_text = transcript.text
                    with st.expander('Transkript'):
                        # Create an expander in the Streamlit application with label 'Koraci'
                        st.info(transcritpt_text)
                        # Display the intermediate steps inside the expander
        if transcritpt_text is not None:
            st.download_button("Download transcript",
                               transcritpt_text, file_name="transcript.txt")


name, authentication_status, username = positive_login(main, "06.09.23")
