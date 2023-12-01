import openai
import speech_recognition as sr
import pyttsx3

engine = pyttsx3.init()

openai.api_key = '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'

def listen_and_respond():
    """
    Listen for audio input, recognize it and respond using OpenAI
    """
    # Create speech recognizer object
    r = sr.Recognizer()

    # Listen for input
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)

    # Try to recognize the audio
    try:
        prompt = r.recognize_google(audio, language="en-EN", show_all=False)
        print("You asked:", prompt)

        # Use OpenAI to create a response
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            max_tokens=300
        )

        # Get the response text
        response_text = str(response['choices'][0]['text']).strip('\n\n')
        print(response_text)

        # Speak the response
        engine.say(response_text)
        engine.runAndWait()
        print()

    # Catch if recognition fails
    except sr.UnknownValueError:
        response_text = "Sorry, I didn't understand what you said"
        print(response_text)
        engine.say(response_text)
        engine.runAndWait()
        print()
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

def ask_openai(question):
    response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=question,
            temperature=0.7,
            max_tokens=300
        )
    return response.choices[0].text.strip()

#print("You can start talking to the AI assistant. Type 'exit' to end the conversation.")

while True:
    listen_and_respond()
    #user_input = input("You: ")
    #if user_input.lower() == 'exit':
        #break
    
    #response = ask_openai(user_input)
    #print("AI: " + response)
