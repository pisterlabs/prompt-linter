import openai
import streamlit as st
import speech_recognition as sr
import pyttsx3

st.title("Walmart Store Assistant")

recognizer = sr.Recognizer()
engine = pyttsx3.init()

messages = [{"role": "system", "content": "You are a Walmart store assistant ready to help with inquiries about the store."}]

openai.api_key = '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'

response = ""

def speak(text):
    engine.say(text)
    engine.runAndWait()

def CustomChatGPT(user_input):
    global response
    messages.append({"role": "user", "content": user_input})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": response})

def voice_interface():
    st.write("Assistant: Welcome to Walmart! How can I assist you today?")
    # speak("Welcome to Walmart! How can I assist you today?")
    
    while True:
        with sr.Microphone() as source:
            st.write("Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, phrase_time_limit=5)
            try:
                user_input = recognizer.recognize_google(audio).lower()
                st.write("You said:", user_input)
                
                if user_input in ["ok done","quit","stop", "bye", "exit", "goodbye"]:
                    st.write("Assistant: Thank you for visiting Walmart. Have a great day!")
                    break
                
                CustomChatGPT(user_input)
                st.write("Assistant:", response)
                speak(response,timeout=1)
            except sr.UnknownValueError:
                st.write("Assistant: Sorry, I didn't understand that.")
            except sr.RequestError:
                st.write("Assistant: Sorry, I'm having trouble connecting to the internet.")

def main():
    st.write("Click the button below to start the Walmart Store Assistant:")
    button_clicked = st.button("Start Assistant")
    
    if button_clicked:
        voice_interface()

if __name__ == '__main__':
    main()


