import openai
import speech_recognition as sr
import pyttsx3

openai.api_key = "sk-"

def generate_response(query):
    response = openai.Completion.create(
        engine="text-davinci-002", 
        prompt=query, 
        max_tokens=500,
        temperature=0.7,
        presence_penalty=0.5,
        frequency_penalty=0.5
    )
    return response.choices[0].text.strip()

def search_path(program):
    # Search for the path of the program in the system
    for root, dirs, files in os.walk("C:\\"):
        if program in files:
            return os.path.join(root, program)
    return None


r = sr.Recognizer()
tts = pyttsx3.init()

while True:
    # Wait for "Hey Hank" to be said
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)
        
    try:
        query = r.recognize_google(audio, language='es-ES')
        if "Hey Hank" in query.lower():
            # Respond to the user
            tts.say("Hi, how can I help you?")
            tts.runAndWait()
            
            while True:
                # Listen to the user input
                with sr.Microphone() as source:
                    print("Speak now:")
                    audio = r.listen(source)
                
                try:
                    input = r.recognize_google(audio, language='en-US')
                    response = generate_response(input)
                    print("Response:", response)
                    tts.say(response)
                    tts.runAndWait()
                    
                    if "goodbye" in input.lower():
                        # End the conversation
                        tts.say("Goodbye")
                        tts.runAndWait()
                        break
                        
                except sr.UnknownValueError:
                    print("Could not understand what you said")
                except sr.RequestError as e:
                    print("Error connecting to speech recognition service; {0}".format(e))
                    
        else:
            print("Keyword not detected")
            
    except sr.UnknownValueError:
        print("Could not understand what you said")
    except sr.RequestError as e:
        print("Error connecting to speech recognition service; {0}".format(e))
