
import openai
import speech_recognition as sr
import configparser
import pyttsx3
import re
import webbrowser
import time
import pywhatkit
import sys
import json
import platform
import subprocess
import os



art = r'''
$$$$$$$\ $$\     $$\ $$\       $$$$$$$$\ $$\   $$\  $$$$$$\
$$  __$$\\$$\   $$  |$$ |      $$  _____|$$ |  $$ |$$  __$$\
$$ /  \__|\$$\ $$  / $$ |      $$ |      \$$\ $$  |$$ /  $$ |
$$ |       \$$$$  /  $$ |      $$$$$\     \$$$$  / $$$$$$$$ |
$$ |        \$$  /   $$ |      $$  __|    $$  $$<  $$  __$$ |
$$ |  $$\    $$ |    $$ |      $$ |      $$  /\$$\ $$ |  $$ |
\$$$$$$  |   $$ |    $$$$$$$$\ $$$$$$$$\ $$ /  $$ |$$ |  $$ |
 \______/    \__|    \________|\________|\__|  \__|\__|  \__|

 Created By Cycus Pectus
 Twitter: @_cytech
 Your Best Ai Virtual Assistant
'''

print(art)

# Load the configuration file
config = configparser.ConfigParser()
config.read('config.ini')

openai.api_key = config.get('API', 'key')
MODEL_ENGINE = "text-davinci-003"
AI = "CYLEXA"  # Updated assistant name

class ChatbotAssistant(object):
    def __init__(self):
        self.speech_recognizer = sr.Recognizer()
        self.conversation_history = ""
        self.username = ""
        self.silent = False
        self.sleep_timeout = None
        self.wake_words = ["alexa", "cylexa"]
        self.sleep_duration = 60  # 1 minute in seconds
        self.last_activity_time = time.time()

    def speak_text(self, text, speed=50):
        try:
            engine = pyttsx3.init()

            # Configure the voice
            voices = engine.getProperty('voices')
            engine.setProperty('voice', voices[1].id)  # Change the index to select a different voice

            rate = engine.getProperty('rate')
            engine.setProperty('rate', rate - speed)

            engine.say(text)
            engine.runAndWait()
        except:
            return False
        return True

    def generate_response(self, prompt):
        response = openai.Completion.create(
            model=MODEL_ENGINE,
            prompt=prompt,
            temperature=0.5,
            max_tokens=1024,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )

        return response.choices[0].text

    def get_conversation(self, prompt):
        prompt = self.conversation_history + prompt

        response = self.generate_response(prompt)

        response_parts = response.split("::", 1)
        if "::" in response:
            response = response_parts[1]
        else:
            response = response_parts[0]
        self.conversation_history += f"{AI}:: {response} "

        return response

    def extract_name(self, user_input):
        # Use regular expressions to extract the name from user input
        pattern = r"my name is ([A-Za-z]+)"
        match = re.search(pattern, user_input)
        if match:
            return match.group(1)
        else:
            return None

    def play_music(self):
        self.speak_text("Sure, what song would you like to listen to?")
        while True:
            try:
                with sr.Microphone() as source:
                    print(f"{AI}: What song would you like to listen to?")
                    self.speech_recognizer.adjust_for_ambient_noise(source)

                    # Adjust the energy threshold to control sensitivity
                    self.speech_recognizer.energy_threshold = 1000

                    audio = self.speech_recognizer.listen(source)

                    # Perform speech recognition on the captured audio
                    user_input = self.speech_recognizer.recognize_google(audio)
                    user_input = user_input.lower().strip()

                    # Play the song
                    pywhatkit.playonyt(user_input)

                    # Extract the name of the song from the user input
                    song_name = user_input

                    response = f"Playing {song_name}. Enjoy the music!"
                    self.speak_text(response)
                    break

            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                self.speak_text("Didn't get you clear!")
                self.speak_text("Please say it again.")

            except sr.UnknownValueError:
                print("Unknown error occurred!")
                self.speak_text("Didn't get you clear!")
                self.speak_text("Please say it again.")


    def search_and_play(self):
        self.speak_text("Sure, Which video would you like to watch?")
        while True:
            try:
                with sr.Microphone() as source:
                    print(f"{AI}: Which video would you like to watch?")
                    self.speech_recognizer.adjust_for_ambient_noise(source)

                    # Adjust the energy threshold to control sensitivity
                    self.speech_recognizer.energy_threshold = 1000

                    audio = self.speech_recognizer.listen(source)

                    # Perform speech recognition on the captured audio
                    user_input = self.speech_recognizer.recognize_google(audio)
                    user_input = user_input.lower().strip()

                    # Play the song
                    pywhatkit.playonyt(user_input)

                    # Extract the name of the song from the user input
                    video_name = user_input

                    response = f"Playing {video_name}. Enjoy the video!"
                    self.speak_text(response)
                    break

            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                self.speak_text("Didn't get you clear!")
                self.speak_text("Please say it again.")

            except sr.UnknownValueError:
                print("Unknown error occurred!")
                self.speak_text("Didn't get you clear!")
                self.speak_text("Please say it again.")

                       

            
    def search_youtube(self, query):
        url = f"https://www.youtube.com/results?search_query={query}"
        webbrowser.open(url)
        self.speak_text(f"Here are the YouTube search results for {query}.")
        self.speak_text("How else can I help you?")

    def search_google(self, query):
        url = f"https://www.google.com/search?q={query}"
        webbrowser.open(url)
        self.speak_text(f"Here are the Google search results for {query}.")
        self.speak_text("How else can I help you?")

    def get_username(self):
        self.username = "Friend"

    def create_reminders_file(self):
        try:
            with open("reminders.json", "x") as file:
                json.dump([], file)
        except FileExistsError:
            pass

    def add_reminder(self, date, name, description):
        try:
            with open("reminders.json", "r") as file:
                reminders = json.load(file)
        except FileNotFoundError:
            reminders = []

        reminders.append({
            "date": date,
            "name": name,
            "description": description
        })

        with open("reminders.json", "w") as file:
            json.dump(reminders, file)

    def get_reminders(self):
        try:
            with open("reminders.json", "r") as file:
                reminders = json.load(file)

            if reminders:
                self.speak_text("Here are your reminders:")
                for reminder in reminders:
                    self.speak_text(f"Reminder name: {reminder['name']}")
                    self.speak_text(f"Date Due: {reminder['date']}")
                    self.speak_text(f"Reminder Description: {reminder['description']}")
            else:
                self.speak_text("You don't have any reminders.")

        except FileNotFoundError:
            self.speak_text("You don't have any reminders.")

    def open_application(self, application_name):
        if platform.system() == "Windows":
            subprocess.Popen(application_name)
        elif platform.system() == "Linux":
            subprocess.Popen(["xdg-open", application_name])

  

    def shutdown_computer(self):
        if platform.system() == "Windows":
            subprocess.Popen("shutdown /s /t 0", shell=True)
        elif platform.system() == "Linux":
            subprocess.Popen("sudo shutdown now", shell=True)

    def restart_computer(self):
        if platform.system() == "Windows":
            subprocess.Popen("shutdown /r /t 0", shell=True)
        elif platform.system() == "Linux":
            subprocess.Popen("sudo reboot now", shell=True)


    def close_browser(self):
        current_platform = platform.system()

        if current_platform == "Windows":
            # Close Chrome browser
            os.system("taskkill /im chrome.exe /f")

            # Close Firefox browser
            os.system("taskkill /im firefox.exe /f")

            # Close Edge browser
            os.system("taskkill /im msedge.exe /f")

            # Close Internet Explorer browser
            os.system("taskkill /im iexplore.exe /f")

        elif current_platform == "Darwin":
            # Close Chrome browser
            os.system("pkill -f 'Google Chrome'")

            # Close Firefox browser
            os.system("pkill -f 'firefox'")

            # Close Safari browser
            os.system("pkill -f 'Safari'")

        elif current_platform == "Linux":
            # Close Chrome browser
            os.system("pkill -f 'chrome'")

            # Close Chrome browser
            os.system("pkill -f ' google chrome'")

            # Close Firefox browser
            os.system("pkill -f 'firefox'")

            # Close Chromium browser
            os.system("pkill -f 'chromium'")

    
    def open_default_browser(self):
        system = platform.system()
        url = 'https://www.google.com'  

        if system == 'Windows':
            webbrowser.open(url)
        elif system == 'Linux':
            try:
                webbrowser.get('firefox').open(url)  # Try opening with Firefox
            except webbrowser.Error:
                webbrowser.open(url)  # If Firefox is not available, open with the default browser
        else:
            print("Unsupported operating system")

    def run(self):
        self.get_username()
        self.create_reminders_file()

        silent_timeout = time.time()  # Variable to track the silent timeout
        self.speak_text(f"Hello {self.username}! Welcome to CYLEXA - Your AI Virtual Assistant! I am  an intelligent virtual assistant that can assist you with various tasks and provide realtime helpful and educational information, thanks to open ai. I can also  play music, search  and play YouTube videos and do Google searches for you, i can set reminders and present them to you anytime you want me to remind you, i also can list all running processes on your computer, and even shut down or restart your computer whenever you wish. I support both Windows and Linux operating systems, ensuring a seamless experience regardless of your system. Now ask me anything and i am ready to answer or help.")
        while True:
            try:
                with sr.Microphone() as source:
                    self.speech_recognizer.adjust_for_ambient_noise(source)
                    audio = self.speech_recognizer.listen(source)
                    user_input = self.speech_recognizer.recognize_google(audio)

                    user_input = user_input.lower().strip()
                    self.conversation_history += f"{self.username}:: {user_input}. "

                    if "exit cylexa" in user_input or "quit cylexa" in user_input or "exit alexa" in user_input or "quit alexa" in user_input:
                        self.speak_text("Goodbye!")
                        sys.exit()

                    # Process user input and get the response
                    if "play music" in user_input or "sing a song" in user_input or "sing me a song" in user_input or "play some music" in user_input or "some good music" in user_input or "sing for me" in user_input or "play another song" in user_input:
                        self.play_music()
                        response = "Enjoy the music!"
                    elif "search youtube and" in user_input or "youtube video" in user_input or "search a video" in user_input or "search for a video" in user_input or "open youtube" in user_input or "play on youtube" in user_input or "play video" in user_input or "play a video" in user_input or "play in youtube" in user_input or "watch on youtube" in user_input or "play another video" in user_input or "video on youtube" in user_input or "video in youtube" in user_input:
                        self.search_and_play()
                        response = "Enjoy the video!"
                    elif "stop video" in user_input or "stop a video" in user_input or "stop the video" in user_input or "stop music" in user_input or "close browser" in user_input or "close the browser" in user_input or "exit the browser" in user_input or "cancel song" in user_input or "stop playing" in user_input or "cancel video" in user_input or "not the song" in user_input or "not the video" in user_input or "stop song" in user_input or "stop the song" in user_input or "stop this song" in user_input or "stop this video" in user_input or "stop the video" in user_input or "stop browser" in user_input or "stop my browser" in user_input or "close my browser" in user_input:
                        self.close_browser()
                        stopping = "Okay, successfully stopped"
                        self.speak_text(stopping)
                    elif "open browser" in user_input or "open my browser" in user_input or "open the browser" in user_input or "open a browser" in user_input or "to search the internet" in user_input:
                        self.open_default_browser()
                        open_bro = "Sure,  browser opened"
                        self.speak_text(open_bro)
                    elif "search youtube for" in user_input or "youtube search" in user_input:
                        query = user_input.replace("search youtube for", "").strip()
                        self.search_youtube(query)
                    elif "search google for" in user_input or "google for me" in user_input or "google search" in user_input or "search in google" in user_input:
                        query = user_input.replace("search google for", "").strip()
                        self.search_google(query)
                    elif "set reminder" in user_input or "set a reminder" in user_input or "set reminders" in user_input or "set my reminder" in user_input:
                        self.speak_text("Please provide the reminder name.")
                        audio = self.speech_recognizer.listen(source)
                        name = self.speech_recognizer.recognize_google(audio)
                        name = name.strip()

                        self.speak_text("Please provide the date due.")
                        audio = self.speech_recognizer.listen(source)
                        date = self.speech_recognizer.recognize_google(audio)
                        date = date.strip()

                        self.speak_text("Please provide the description.")
                        audio = self.speech_recognizer.listen(source)
                        description = self.speech_recognizer.recognize_google(audio)
                        description = description.strip()

                        self.add_reminder(date, name, description)
                        self.speak_text("Reminder added successfully.")
                    elif "inquire reminder" in user_input or "inquire reminders" in user_input or "enquire reminder" in user_input or "enquire reminders" in user_input or "list reminders" in user_input or "list reminder" in user_input or "reminders" in user_input:
                        self.get_reminders()
                    elif "restart my computer" in user_input or "restart this computer" in user_input or "reboot my" in user_input or "reboot this" in user_input:
                        self.restart_computer()
                    elif "list processes" in user_input or "list process" in user_input or "list my processes" in user_input or "list all processes" in user_input:
                        self.list_processes()
                
                    else:
                        # Ask GPT-3 model for a response
                        response = self.generate_response(user_input)
                        print(response)
                        self.speak_text(response)

            except sr.UnknownValueError:
                pass
            except sr.RequestError:
                self.speak_text("I'm sorry, but I couldn't access the speech recognition service. Please check your internet connection.")

    # Rest of the code...


if __name__ == "__main__":
    assistant = ChatbotAssistant()
    assistant.run()
