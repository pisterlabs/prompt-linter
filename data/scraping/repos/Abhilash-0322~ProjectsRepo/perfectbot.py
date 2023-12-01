import speech_recognition as sr
from googlesearch import search
import win32com.client
import webbrowser
import datetime
import openai
import subprocess
import requests
import time

drink_water_timer = time.time()
class VoiceAssistant:
    def __init__(self):
        self.speaker = win32com.client.Dispatch("SAPI.SpVoice")        
        openai.api_key = "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"
    
    # def handle_drink_water_reminder(self):
    #     current_time = time.time()
    #     if current_time - self.drink_water_timer >= 30 * 60:  # 30 minutes in seconds
    #         self.speaker.Speak("It's time to drink some water. Stay hydrated!")
    #         self.drink_water_timer = current_time  # Reset the timer

    def get_weather_info(self, location):
        api_key = "378f3248f68a4674b5e112941231110"  # Replace with your WeatherAPI API key
        weather_url = f"https://api.weatherapi.com/v1/current.json?key={api_key}&q={location}"

        try:
            response = requests.get(weather_url)
            if response.status_code == 200:
                weather_data = response.json()
                temperature = weather_data["current"]["temp_c"]
                condition = weather_data["current"]["condition"]["text"]
                humidity = weather_data["current"]["humidity"]
                return f"The weather in {location} is {condition}. Temperature: {temperature}°C. Humidity: {humidity}%."
            else:
                return "Sorry, I couldn't fetch the weather data at the moment."
        except Exception as e:
            print(f"Error fetching weather data: {str(e)}")
            return "Sorry, I encountered an error while fetching weather data."

    def gptfunction(self, query):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": "act as mentor"
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

    def take_command(self):
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
                print("What do you mean? Say that again!")
                return "What Do You Mean, say again"
# from googlesearch import search

class WebInteraction:
    @staticmethod
    def perform_google_search(query):
        try:
            search_results = list(search(query, num=5, stop=5, pause=2))
            return search_results
        except Exception as e:
            print(f"Error performing Google search: {str(e)}")
            return []
    
    def open_website(self, url):
        webbrowser.open(url)

class ApplicationInteraction:
    def open_application(self, app_path):
        try:
            subprocess.Popen(app_path)
            print(f"Opening {app_path}")
        except Exception as e:
            print(f"Error opening {app_path}: {str(e)}")

class SpecificTasks:
    def __init__(self,speaker):
        self.drink_water_timer = time.time()

    def get_current_time(self):
        strfTime = datetime.datetime.now().strftime("%H:%M:%S")
        print(strfTime)
        speaker.Speak(f"Current time is {strfTime}")

    def handle_drink_water_reminder(self):
        current_time = time.time()
        if current_time - drink_water_timer >= 30 * 60:  # 30 minutes in seconds
            speaker.Speak("It's time to drink some water. Stay hydrated!")
            self.drink_water_timer = current_time  # Reset the timer

class FileHandling:
    def write_to_file(self, filename, data):
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(data + '\n')

class ChatbotControl:
    def __init__(self, speaker):
        self.speaker = speaker  # Pass the speaker instance
        self.count = 0
            
    def start(self):
        while True:
            print("listening....")
            query = voice_assistant.take_command()
            
            if "What Do You Mean, say again".lower() in query.lower():
                if self.count<1:
                    speaker.speak("What do you mean?! Say that again!")
                else:
                    print("Be Clear this Time!")
                    speaker.speak("What do you mean?! Say that again! Be clear this time!")
                self.count+=1
                continue

            if "Search for".lower() in query.lower():
                search_query = query.lower().replace("search for", "").strip()
                search_results = web_interaction.perform_google_search(search_query)

                if search_results:
                    responses = f"Here are some search results for '{search_query}':\n"
                    for i, result in enumerate(search_results, start=1):
                        responses += f"{i}. {result}\n"
                else:
                    responses = "I couldn't find any search results for that query."
            else:
                responses = voice_assistant.gptfunction(query)

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
                        application_interaction.open_application(item[1])
                    else:
                        web_interaction.open_website(item[1])
                        continue
                 
                 # Play a specific YouTube video
            if "start Coding Music".lower() in query.lower():
                my_url='https://www.youtube.com/watch?v=4cEKAYnxbrk&t=5324s'
                webbrowser.open(my_url)
                song_name = "Playing Coding Music"
                speaker.speak("Let's Code Together")
                print(f"Playing {song_name}")
                continue
                
            if "play YouTube video".lower() in query.lower():
                # Replace 'VIDEO_URL' with the URL of the YouTube video you want to play
                video_url = 'https://www.youtube.com/watch?v=gV8RL2xcmaQ&list=RDgV8RL2xcmaQ&start_radio=1'
                webbrowser.open(video_url)
                song_name = "Where we started"
                speaker.speak(f"playing {song_name}")
                print(f"Playing {song_name}")
                continue

            # Handle specific tasks
            if "the time" in query:
                specific_tasks.get_current_time()
                continue

            # Modify responses based on specific queries
            if "weather" in query.lower():
                location = query.split("in")[1].strip()
                weather_response = voice_assistant.get_weather_info(location)
                print(weather_response)
                self.speaker.Speak(weather_response)
                continue

            # Handle drink water reminder
            specific_tasks.handle_drink_water_reminder()

            # Write responses to a file
            file_handling.write_to_file('girlai.txt', responses)

            print(f"AI response: {responses}")
            speaker.Speak(responses)

if __name__ == "__main__":
    speaker = win32com.client.Dispatch("SAPI.SpVoice")  # Initialize the speaker
    voice_assistant = VoiceAssistant()
    web_interaction = WebInteraction()
    application_interaction = ApplicationInteraction()
    specific_tasks = SpecificTasks(speaker)  # Pass the speaker instance
    file_handling = FileHandling()
    chatbot_control = ChatbotControl(speaker)  # Pass the speaker instance
    chatbot_control.start()