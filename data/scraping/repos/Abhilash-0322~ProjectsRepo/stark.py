import os
import speech_recognition as sr
from googlesearch import search
import win32com.client
import webbrowser
import datetime
import openai
import subprocess
import time
import pygetwindow as gw

speaker = win32com.client.Dispatch("SAPI.SpVoice")
openai.api_key = "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"
drink_water_timer = time.time()  # time initialized
created_name=datetime.datetime.now().strftime("%D%H%M%S")
def gptfunction(query):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Im your Mentor"
            },
            {
                "role": "user",
                "content": query
            }
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    generated_text = response["choices"][0]["message"]["content"]

    return generated_text

def perform_google_search(query):
    try:
        search_results = list(search(query, num=5, stop=5, pause=2))
        return search_results
    except Exception as e:
        return str(e)

def create_python_file(file_name, content):
    with open(file_name, 'w') as f:
        f.write(content)

        # Bring the file to the front
    try:
        # Change 'YourIDE' to the title of your IDE window
        ide_window = gw.getWindowsWithTitle('Visual Studio Code')[0]
        ide_window.activate()
    except IndexError:
        print("IDE window not found. Make sure to specify the correct window title.")



def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.pause_threshold = 0.6
        audio = r.listen(source)
        try:
            print('recognizing')
            query = r.recognize_google(audio, language='en-us' and 'en-in')
            print(f"User said: {query}")
            return query
        except Exception as e:
            print("What Do You Mean, say again")
            return "What Do You Mean, say again"

while True:
    print("listening....")
    query = takeCommand()

    if "search for".lower() in query.lower():
        search_query = query.lower().replace("search for", "").strip()
        search_results = perform_google_search(search_query)

        if search_results:
            responses = f"Here are some search results for '{search_query}':\n"
            for i, result in enumerate(search_results, start=1):
                responses += f"{i}. {result}\n"
        else:
            responses = "I couldn't find any search results for that query."
    else:
        # Your existing code for other functionalities
        responses = gptfunction(query)

    # Add more websites and applications to open
    sites_and_apps = [
        ["youtube", "https://youtube.com"],
        ["wikipedia", "https://www.wikipedia.org"],
        ["gmail", "https://www.gmail.com"],
        ["github", "https://www.github.com"],
        ["notepad", "notepad.exe"],  # Example of opening Notepad
        ["calculator", "calc.exe"]  # Example of opening Calculator
    ]

    for item in sites_and_apps:
        if f"open {item[0]}".lower() in query.lower():
            if item[1].endswith(".exe"):
                # If it's an application, use subprocess to open it
                try:
                    subprocess.Popen(item[1])
                    print(f"Opening {item[0]}")
                except Exception as e:
                    print(f"Error opening {item[0]}: {str(e)}")
            else:
                # If it's a website, open it in the web browser
                webbrowser.open(f"{item[1]}")
                print(f"Opening {item[0]}")

    # Play a specific YouTube video
    if "play YouTube video".lower() in query.lower():
        # Replace 'VIDEO_URL' with the URL of the YouTube video you want to play
        video_url = 'https://www.youtube.com/watch?v=gV8RL2xcmaQ&list=RDgV8RL2xcmaQ&start_radio=1'
        webbrowser.open(video_url)
        song_name = "Where we started"
        speaker.speak(f"playing {song_name}")
        print(f"Playing {song_name}")

    # Handle specific tasks
    if "the time" in query:
        strfTime = datetime.datetime.now().strftime("%H:%M:%S")
        print(strfTime)
        speaker.Speak(f"Current time is {strfTime}")

    # Modify responses based on specific queries
    if "weather" in query:
        speaker.Speak("I'm sorry, I can't provide weather information at the moment.")

    # Check if it's time for the drink water reminder (every 30 minutes)
    current_time = time.time()
    if current_time - drink_water_timer >= 30 * 60:  # 30 minutes in seconds
        speaker.Speak("It's time to drink some water. Stay hydrated!")
        drink_water_timer = current_time  # Reset the timer

    if "create Python file".lower() in query.lower():
        # You can customize the file name and content as needed
        file_name = f"PythonFile{created_name}.py"
        file_content = "# This is a Python file created by your assistant.\nprint('Hello, World!')"
        create_python_file(file_name, file_content)
        responses = f"I've created a Python file named '{file_name}' in the working directory."

    with open('girlai.txt', 'a', encoding='utf-8') as f:
        g = responses
        f.writelines(g)
        f.writelines('\n')

    print(f"AI response: {responses}")
    speaker.Speak(responses)