import openai
import whisper
import streamlit as st
import os

openai.api_key = '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'
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


# Streamlit functionality
st.title('Whisper ChatGPT Audio')
uploaded_file = st.file_uploader('Choose an audio', accept_multiple_files=False)

if uploaded_file is not None:
    # Save the file to the current working directory
    file_path = os.path.join(os.getcwd(), uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.read())

    with st.spinner(text='Analyzing audio'):
        # Get the content of the uploaded file
        transcription = transcribe_audio(model, file_path)
        summary = CustomChatGPT(transcription)

    st.header('Transcription')
    st.write(transcription)
    st.header('Summary')
    st.write(summary)

    # Remove the file after processing
    os.remove(file_path)
else:
    st.warning('Please upload an audio file.')
