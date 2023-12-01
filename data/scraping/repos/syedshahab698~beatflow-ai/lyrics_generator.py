import streamlit as st
import openai
from pydub import AudioSegment
import os
# import pyttsx3
from gtts import gTTS
# from pyngrok import ngrok



# Streamlit app configuration
st.title("AI Rapper Application")

def generate_lyrics(scenario):
    # OpenAI GPT-3 text generation call
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"You are eminem. write a rap song about {scenario}",
        max_tokens=150,
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    output = response.choices[0].text.strip()
    # remove Verse 1, Verse 2, chorus.
    output = output.replace("Verse 1", "").replace("Verse 2", "").replace("Chorus", "").replace("Verse 3", "")
    
    return output




def generate_rap_song(scenario, background_music_path):
    # Load background music
    background_music = AudioSegment.from_file(background_music_path, format="mp3")
    # Split the background music to get the first 5 seconds
    background_music_5s = background_music[:11000]
    background_music = background_music - 15
    # Generate rap lyrics
    st.write("Generating rap lyrics...")
    lyrics = generate_lyrics(scenario)
    st.write("Lyrics generated:")
    st.write(lyrics)

    # Convert lyrics to MP3
    lyrics_tts = gTTS(lyrics, lang="en-uk", slow=False)
    lyrics_mp3 = "lyrics.mp3"
    lyrics_tts.save(lyrics_mp3)
    # engine = pyttsx3.init()
    # engine.save_to_file(lyrics, lyrics_mp3)
    # engine.runAndWait()
    

    # Load lyrics audio
    lyrics_audio = AudioSegment.from_file(lyrics_mp3, )

    overlayed_audio = lyrics_audio.overlay(background_music[11000:])

    # Overlay lyrics with background music after 5 seconds
    final_audio = background_music_5s + overlayed_audio

    # Save the final rap song
    rap_song_mp3 = "rap_song.mp3"
    final_audio.export(rap_song_mp3, format="mp3")

    # Clean up the temporary lyrics MP3 file
    os.remove(lyrics_mp3)
    # engine.stop()

    return rap_song_mp3

# Streamlit user interface
st.write("Enter a scenario and let the AI rap a song about it!")
scenario = st.text_input("Enter a scenario:")
if st.button("Rap!"):
    if scenario:
        background_music_path = "background.mp3"  # Replace with the path to your background music

        st.write("Generating the rap song...")
        rap_song_mp3 = generate_rap_song(scenario, background_music_path)
        st.write("Rap Song MP3 generated!")

        st.audio(rap_song_mp3, format="audio/mp3")
    else:
        st.write("Please enter a scenario to continue.")


# Setup a tunnel to the streamlit port 8501
# public_url = ngrok.connect(port='8501')
# st.write("Public URL:", public_url)
