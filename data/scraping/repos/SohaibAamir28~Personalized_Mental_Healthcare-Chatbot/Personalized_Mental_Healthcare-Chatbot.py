import openai
import pandas as pd
import speech_recognition as sr
import streamlit as st

# Set OpenAI API key
openai.api_key = "API_KEY"

# Load mental health dataset
mentalhealth = pd.read_csv("AI_Mental_Health.csv")

# Preprocess data
input_text = []
for i in range(len(mentalhealth)):
    input_text.append(mentalhealth.iloc[i]["Questions"])

# Initialize speech recognizer
r = sr.Recognizer()

# Define callback function for voice search
def voice_search_callback(recognizer, audio):
    try:
        query = recognizer.recognize_google(audio)
        st.write("You said: " + query)
        generate_response(query)
    except sr.UnknownValueError:
        st.write("Could not understand audio")
    except sr.RequestError as e:
        st.write("Error: {0}".format(e))

# Define function for generating responses
def generate_response(input_text):
    # Generate responses using Chat GPT
    response = openai.Completion.create(
        engine="davinci",
        prompt=input_text,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.7
    )

    # Print responses
    for i in range(len(response["choices"])):
        st.write(response["choices"][i]["text"])

# Define main function
def main():
    st.title("Personalized Mental HealthCare Chatbot App")
    st.text("\nBy: Sohaib Aamir, Team: AI-HACKERS")
    input_type = st.radio("Select input type", ("Text", "Voice"))
    
    if input_type == "Text":
        input_text = st.text_input("Enter your message")
        input_type += "Regarding Personalized Mental HealthCare Tips"
        if st.button("Send"):
            generate_response(input_text)
    elif input_type == "Voice":
        with sr.Microphone() as source:
            st.write("Speak now...")
            r.adjust_for_ambient_noise(source)
            stop_listening = r.listen_in_background(source, voice_search_callback)

            # Stop the loop when the user presses the "stop" button
            st.write("Press the button below to stop voice search...")
            stop_button = st.button("Stop voice search")
            if stop_button:
                stop_listening()
           
            
if __name__ == '__main__':
    main()



