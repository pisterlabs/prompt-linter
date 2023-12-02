import webbrowser
import datetime
import pyttsx3
import pywhatkit
import speech_recognition as sr
import wikipedia
import pyjokes
import pandas
import pyaudio
import tkinter as tk
import openai

listener = sr.Recognizer()
engine = pyttsx3.init()

openai.api_key = "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"

root = tk.Tk()
root.title("Virtual Assistant")
root.geometry("4000x4000")

# create a label for the output
output_label = tk.Label(root, text="I am your virtual assistant. What can I do for you?", font=("Arial", 44), wraplength=1000, justify="center")
output_label.pack(pady=20)

# create a function to activate the virtual assistant
def activate_assistant():
    button.config(text="Listening...", bg="green", state="disabled")
    command = take_command()
    print(command)
    button.config(text="Activate Assistant", bg="SystemButtonFace", state="normal")

# create a function to get voice command from user
def take_command():
    try:
        with sr.Microphone() as source:
            print('listening...')
            button.config(text="Listening...", bg="green", state="disabled")
            voice = listener.listen(source)
            command = listener.recognize_google(voice)
            command = command.lower()
            if 'alexa' in command:
                command = command.replace('alexa', '')
                print(command)
                
            if 'play' in command:
                song = command.replace('play', '')
                talk('playing ' + song)
                pywhatkit.playonyt(song)
                
            elif 'link' in command:
                query = command.replace('link', '')
                talk(f"Here's the link for {query}")
                search_result = pywhatkit.search(query)
                webbrowser.open(search_result)
            elif 'time' in command:
                time = datetime.datetime.now().strftime('%I:%M %p')
                talk('Current time is ' + time)
            elif 'who is' in command:
                person = command.replace('who is', '')
                info = wikipedia.summary(person, 1)
                print(info)
                talk(info)
            elif 'date' in command:
                talk('sorry, I have a headache')
            elif 'are you single' in command:
                talk('I am in a relationship with wifi')
            elif 'joke' in command:
                talk(pyjokes.get_joke())
            else:
                # Only call OpenAI if none of the above conditions are met
                response = generate_response(command)
                talk(response)
                button.config(text="Activate Assistant", bg="SystemButtonFace", state="normal")
    except:
        pass
    return command


# create a function to generate response using OpenAI API
def generate_response(prompt):
    completions = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=60,
        n=1,
        stop=None,
        temperature=0.7,
    )
    message = completions.choices[0].text.strip()
    return message

# create a function to output the response from the virtual assistant
def talk(text):
    engine.setProperty('voice', 'com.amazon.tts.beta.en-US-Neural')
    engine.say(text)
    engine.runAndWait()
    output_label.configure(text=text)

# create a button to activate the virtual assistant
button = tk.Button(root, text="Activate Assistant", command=activate_assistant)
button.pack()

root.mainloop()
