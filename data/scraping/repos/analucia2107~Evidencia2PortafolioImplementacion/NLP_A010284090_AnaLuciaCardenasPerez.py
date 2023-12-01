# Ana Lucía Cárdenas Pérez A01284090
# Whisper Summarizer con Streamlit

import streamlit as st
import openai
import ssl
import certifi

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
print(ssl.get_default_verify_paths())

st.subheader("Ana Lucía Cárdenas Pérez")
st.subheader("A01284090")
st.divider()
st.header("Whisper Summarizer")
st.subheader("Análisis de Audio para Transcripción y Resumen Escrito")

OPENAI_API_KEY  = "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"
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def transcribe_audio(file_path):
    with open("MA1.m4a", "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
        )
    return transcript.text
    
def custom_chatgpt(user_input):
    messages = [
        {"role": "system", "content": "You are an office administrator, summarize the text in key points"},
        {"role": "user", "content": user_input}
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    chatgpt_reply = response.choices[0].message.content
    return chatgpt_reply


def main():
    file_path = 'MA1.m4a'
    st.audio(file_path, format="audio")
    
    # Transcribe audio
    st.subheader("Transcription:")
    transcription = transcribe_audio(file_path)
    st.write(transcription)

    # Summarize using ChatGPT
    st.subheader("Summary:")
    summary = custom_chatgpt(transcription)
    st.write(summary)



if __name__ == "__main__":
    main()





