import streamlit as st
import openai
import yt_dlp as youtube_dl
import re
import os
# Define Streamlit app
def main():
    st.markdown(""" 
    <style>
    .big-font {
    font-size:30px !important;
    font-weight:bold
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="big-font">Youtube Video Transcription and Summarization</p>',unsafe_allow_html=True)    
    # Input box for YouTube URL
    youtube_url = st.text_input(
        "Please enter the YouTube URL", 
        "https://www.youtube.com/watch?v=5JK7vjVaIvo")


    video_id = youtube_url.replace('https://www.youtube.com/watch?v=', '')

    
    openai.api_key = os.getenv('OPENAI_API_KEY')

    # Submit button
    if st.button("Submit"):
        with st.spinner("Downloading audio from URL..."):
            ydl_opts = {
                "format": "bestaudio/best[ext=m4a]",
                "extractaudio": True,
                "audioformat": "mp3",
                "outtmpl": f"{video_id}.%(ext)s"
            }
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
                info = ydl.extract_info(youtube_url, download=False)
                duration = info["duration"]
                multiplier = 1 + duration // (60*5) + 1


            with st.spinner("Transcribing audio..."):
                with open(f"{video_id}.webm", "rb") as audio:
                    transcript = openai.Audio.transcribe("whisper-1", audio)
                    transcript = transcript["text"]

        with st.expander("Expand to see the transcript"):
            st.text_area('Transcript', transcript)

        
        # Summarize transcription using OpenAI GPT API
        with st.spinner("Summarising transcript.."):
            summary = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                        {
                            "role": "system", 
                            "content": (
                                "You are a helpful assistant that"
                                " summarises youtube video transcripts into "
                                f"{300*multiplier} words or less."
                                )
                        },
                        {"role": "user", "content": transcript}
                    ]
            )

        # Display summary
        st.write("Summary of the video:", summary.choices[0]["message"]["content"])

if __name__ == '__main__':
    # LOGGED_IN key is defined by streamlit_login_auth_ui in the session state.
    if 'LOGGED_IN' in st.session_state and st.session_state.LOGGED_IN:
        main()
    else:
        st.write("Please login first")