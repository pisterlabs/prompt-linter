
import openai
import whisper
import streamlit as st

openai.api_key = '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'
model = whisper.load_model("base")

st.title("Audio summarize using whisper")
st.subheader("Portfolio - NLP Module")
st.subheader("Jorge de León - A00829759")
st.write('\n\n')

file_path = st.text_input("Enter a the audio file path to analyze","Write Here...")

def transcribe_audio(model, file_path):
    transcript = model.transcribe(file_path)
    return transcript['text']

def CustomChatGPT(user_input):
    messages = [{"role": "system", "content": "You are an office administer, summarize the text in key points"}]
    messages.append({"role": "user", "content": user_input})
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = messages
    )
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    return ChatGPT_reply

if st.button('Predict Sentiment'):
    st.write("Generating summarized text..")
    transcription = transcribe_audio(model, file_path)
    summary = CustomChatGPT(transcription)
    print(summary)
    st.subheader("Summarized text")
    st.write(summary)
else:
	st.write("Press the above button..")



