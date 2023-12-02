import streamlit as st
import openai
import ssl
import certifi


ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
print(ssl.get_default_verify_paths())

st.title("Generador de puntos clave atraves de audio")


OPENAI_API_KEY  = "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"
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def transcribe_audio(file):
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=file)
        
    return transcript.text
    
def custom_chatgpt(user_input):
    messages = [
        {"role": "system", "content": "You are an office administrator, summarize the text in key points. Please give your answer in the language the text is in."},
        {"role": "user", "content": user_input}
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    chatgpt_reply = response.choices[0].message.content
    return chatgpt_reply


uploaded_audio = st.file_uploader("Elige un archivo de audio", accept_multiple_files=False)

if uploaded_audio:
    st.audio(uploaded_audio, format="audio")

    # Transcribe audio
    st.subheader("Transcripción:")
    transcription = transcribe_audio(uploaded_audio)
    st.write(transcription)

    # Summarize using ChatGPT
    st.subheader("Puntos clave:")
    summary = custom_chatgpt(transcription)
    st.write(summary)

