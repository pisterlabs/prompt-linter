import streamlit as st
import openai
import whisper
import tempfile
import os

# Set OpenAI API key
openai.api_key = "API key"

# Load Whisper model
model = whisper.load_model("base")

def transcribe_audio(model, file_path):
    transcript = model.transcribe(file_path)
    return transcript['text']

def CustomChatGPT(user_input):
    messages = [{"role": "system", "content": "You are an office administrator, summarize the text in key points"}]
    messages.append({"role": "user", "content": user_input})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    return ChatGPT_reply

def main():
    st.title("Audio Transcription and Summarization")
    st.subheader("Francisco Castorena, A00827756")

    # File upload
    uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a"])

    if uploaded_file:
        # Display audio controls
        st.audio(uploaded_file, format='audio')

        # Save uploaded file to a temporary location
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())

        # Transcribe audio
        transcription = transcribe_audio(model, temp_file.name)

        # Summarize using ChatGPT
        summary = CustomChatGPT(transcription)

        # Display results
        st.subheader("Transcription:")
        st.write(transcription)

        st.subheader("Summary:")
        st.write(summary)

        # Clean up temporary file
        os.unlink(temp_file.name)

if __name__ == "__main__":
    main()
