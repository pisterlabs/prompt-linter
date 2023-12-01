import pprint
import google.generativeai as palm
import speech_recognition as sr
import openai
import pyttsx3
import webbrowser

recognizer = sr.Recognizer()
microphone = sr.Microphone()

def initialize_text_to_speech():
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[0].id)  # Change voice index as needed
        return engine
    except Exception as e:
        print(f"Error initializing text-to-speech: {str(e)}")
        return None

def speak(text, engine):
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Error speaking text: {str(e)}")

def recognize_speech():
    try:
        with microphone as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        user_input = recognizer.recognize_google(audio).lower()
        print("You:", user_input)
        return user_input
    except sr.UnknownValueError:
        return ""
    except Exception as e:
        print(f"Error recognizing speech: {str(e)}")
        return ""

def generate_text(prompt):
    try:
        palm.configure(api_key='AIzaSyBla3Lqx37-XX1cI53RPjvaKvbR7w2trfQ')
        models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
        model = models[0].name

        completion = palm.generate_text(
            model=model,
            prompt=prompt,
            temperature=0,
            # The maximum length of the response
            max_output_tokens=800,
        )

        return completion.result
    except Exception as e:
        print(f"listening")
        return ""

if __name__ == "__main__":
    text_to_speech_engine = initialize_text_to_speech()
    
    if text_to_speech_engine is None:
        print("Failed to initialize text-to-speech. Exiting.")
    else:
        while True:
            user_input = recognize_speech()
            prompt = user_input
            
            if "quit" in prompt:
                webbrowser.open("C:\\Users\\91630\\Desktop\\Project-V\\PROJECT_V.PY")
                break
            else:
                generated_text = generate_text(prompt)
                print(generated_text)
                
