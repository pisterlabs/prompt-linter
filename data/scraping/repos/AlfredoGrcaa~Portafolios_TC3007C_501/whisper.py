import streamlit as st
import openai
import whisper
import tempfile
import os

# Set OpenAI API key
openai.api_key = "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"

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
    st.title("Actividad NLP: Whisper + ChatGPT")
    st.subheader("José Alfredo García Rodríguez, A00830952")

    # File upload
    uploaded_file = st.file_uploader("Elegir documento", type=["mp3", "wav", "m4a"])

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

        # Show/hide summary button
        show_summary = st.button("Show Summary")
        if show_summary:
            # Styling for Summary
            st.subheader("Summary:")
            st.markdown(f'<p style="font-size:16px; font-family: Arial, sans-serif;">{summary}</p>', unsafe_allow_html=True)

        # Show/hide transcription button
        show_transcription = st.button("Show Transcription")
        if show_transcription:
            # Styling for Transcription
            st.subheader("Transcription:")
            st.markdown(f'<p style="font-size:14px; font-family: "Times New Roman", serif;">{transcription}</p>', unsafe_allow_html=True)

        # Clean up temporary file
        os.unlink(temp_file.name)

if _name_ == "_main_":
    main()