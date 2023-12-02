from gtts import gTTS
from io import BytesIO
from audio_recorder_streamlit import audio_recorder
import streamlit as st
import openai
import speech_recognition as sr

openai.api_key = "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"
recognizer = sr.Recognizer()

def listen(audio_bytes):
    try:
        audio_data = sr.AudioData(audio_bytes, sample_rate=44100, sample_width=4)
        text = recognizer.recognize_google(audio_data, language='th-TH')
        return text
    except sr.UnknownValueError:
        st.write("❌ขออภัย ไม่สามารถรับเสียงของคุณได้ กรุณาพูดใหม่อีกครั้ง")
    except sr.RequestError as e:
        st.write("❌ขออภัย ไม่สามารถรับเสียงของคุณได้ กรุณาพูดใหม่อีกครั้ง: " + e)
    return ""

def generate_response(prompt):
    response = openai.Completion.create(
        engine = "text-davinci-003",
        prompt = f"{prompt}\n\nLanguage: th",
        max_tokens = 500,
        n = 1,
        stop = None,
        temperature = 0.5
    )
    return response.choices[0].text.strip()

def convert_str_to_audio_data(audio_text):
    audio_data = BytesIO()
    tts = gTTS(audio_text, lang = "th")
    tts.write_to_fp(audio_data)

    return audio_data

def ai_assistant(audio_bytes):
    st.divider()
    speech = listen(audio_bytes)
    if speech == "":
        response = "ขออภัย ไม่สามารถรับเสียงของคุณได้ กรุณาพูดใหม่อีกครั้ง"
        audio_data = convert_str_to_audio_data(response)

        st.subheader("เสียง🔊")
        st.audio(audio_data)
    else:
        if "ลาก่อน" in speech:
            st.write("You said: " + speech)
            response = "ขอบคุณที่ใช้บริการค่ะ"
            st.write("AI said: " + response)

            audio_data = convert_str_to_audio_data(response)
            st.subheader("เสียง🔊")
            st.audio(audio_data)
        else:
            st.subheader("บทสนทนา💬")
            st.write("คุณ: {speech}".format(speech=speech))
            response = generate_response(speech)
            st.write("ผู้ช่วยอัจฉริยะ: {response}".format(response=response))

            audio_data = convert_str_to_audio_data(response)
            st.subheader("เสียง🔊")
            st.audio(audio_data)

st.header("Smart Assistant💡")
st.divider()
st.subheader("กดปุ่มเพื่อรับเสียงพูด🎤")
audio_bytes = audio_recorder(text="", icon_size="10x")
if audio_bytes:
    ai_assistant(audio_bytes)