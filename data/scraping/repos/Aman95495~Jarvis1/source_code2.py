import os
import subprocess
import win32com.client
import speech_recognition as sr
import webbrowser
import openai
import random
import datetime
import wikipedia
from config import apikey
from config import weather
import requests


# This help jarvis to speak
speaker = win32com.client.Dispatch("SAPI.SpVoice")

# Giving Command i.e this function covert speech command to text and return it.
def takeCommand():
    r=sr.Recognizer()
    with sr.Microphone() as source:
        r.pause_threshold = 0.6
        r.energy_threshold = 300
        audio  = r.listen(source)
        try:
            print("Recognizing....")
            query = r.recognize_google(audio, language="en-in")
            print(f'User Said: {query}')
            return query
        except  Exception as e:
            return "Sorry Sir Some Error Occured."
        
# calling openai function
def ai(prompt):
    openai.api_key = apikey

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    try:
        return response.choices[0].text
    except Exception as e:
        return str(e)
    
# asking for weather details
def askweather(location):
    # Replace with your OpenWeatherMap API key
    api_key = weather

    # Define the parameters for your request (e.g., city name and units)
    params = {
        "q": location,         # Replace with the desired location (city, country)
        "units": "metric",      # Use "imperial" for Fahrenheit
        "appid": api_key
    }

    # Define the base API URL for the "weather" endpoint
    base_url = "https://api.openweathermap.org/data/2.5/weather"

    # Construct the complete URL by combining the base URL and parameters
    api_url = f"{base_url}?q={params['q']}&units={params['units']}&appid={params['appid']}"

    # Send the GET request
    response = requests.get(api_url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        data = response.json()
        # Print the weather data
        '''
        print("City:", data["name"])
        print("Temperature:", data["main"]["temp"], "°C")
        print("Description:", data["weather"][0]["description"])
        '''
        speaker.speak(f"Sir, in {data['name']}, today's temperature is {data['main']['temp']} degrees Celsius with {data['weather'][0]['description']} conditions.")

    else:
        print("Error:", response.status_code)
        speaker.speak("Sorry Sir Some Error Occurred!")

# Wish Function
def wishMe():
    hour=int(datetime.datetime.now().hour)
    if 0 <= hour < 4:
        speaker.speak("Good night Sir!")
    elif 4 <= hour < 12:
        speaker.speak("Good morning Sir!")
    elif 12 <= hour < 16:
        speaker.speak("Good afternoon Sir!")
    else:
        speaker.speak("Good evening Sir!")
 


# Main Function
if __name__ == '__main__':
    wishMe()
    speaker.speak(" I am Jarvis, How may i help you?")

    while True:
        print("Listening....")
    
        query = input("command :: ")
        print(f"User Said :: {query}")

        #### For Opening Different Sites Using Command "open <site name>"
        sites = [
            ["youtube", "https://www.youtube.com"],
            ["google", "https://www.google.com"],
            ["facebook", "https://www.facebook.com"],
            ["twitter", "https://www.twitter.com"],
            ["leetcode", "https://leetcode.com"],
            ["hackerearth", "https://www.hackerearth.com"],
            ["Wikipedia", "https://www.wikipedia.org"],
            ["Tinkercad", "https://www.tinkercad.com"],
            ["LinkedIn", "https://www.linkedin.com"],
            ["AngelList", "https://www.angel.co"],
            ["Google Scholar", "https://scholar.google.com"],
            ["Coursera", "https://www.coursera.org"],
            ["edX", "https://www.edx.org"],
            ["Khan Academy", "https://www.khanacademy.org"],
            ["MIT OpenCourseWare", "https://ocw.mit.edu"],
            ["Harvard Online Courses", "https://online-learning.harvard.edu"],
            ["Stanford Online", "https://online.stanford.edu"],
            ["Udacity", "https://www.udacity.com"],
            ["Codecademy", "https://www.codecademy.com"],
            ["Duolingo", "https://www.duolingo.com"],
            ["TED Talks", "https://www.ted.com"],
            ["National Geographic Kids", "https://kids.nationalgeographic.com"],
            ["NASA", "https://www.nasa.gov"],
            ["Smithsonian Institution", "https://www.si.edu"],
            ["History.com", "https://www.history.com"],
            ["Discovery Channel", "https://www.discovery.com"],
            ["Britannica", "https://www.britannica.com"],
            ["OpenStax", "https://openstax.org"],
            ["Project Gutenberg", "https://www.gutenberg.org"],
            ["SparkNotes", "https://www.sparknotes.com"],
            ["Chemguide", "http://www.chemguide.co.uk"],
            ["Geology.com", "https://geology.com"],
            ["Internet Archive", "https://archive.org"],
            ["National Archives", "https://www.archives.gov"],
            ["Smithsonian Learning Lab", "https://learninglab.si.edu"]
        ]

        for site in sites:
            if f'open {site[0]}'.lower() in query:
                speaker.speak(f'Opening {site[0]} sir...')
                webbrowser.open(site[1])

        
        ### Music 
        if "open music" in query:
            musicPath = "C:\\Users\\DELL\\Music"
            songs = os.listdir(musicPath)
            print(songs)
            if songs:
                speaker.speak("Opening Music Sir!")
                random_song = random.choice(songs)
                song_path = os.path.join(musicPath, random_song)
                
                try:
                    # Use subprocess to open the music file in the background
                    subprocess.Popen(['start', '', song_path], shell=True)
                except Exception as e:
                    print(f"Error: {e}")
            else:
                speaker.speak("Sorry Sir Can't Play Music")

        elif 'search google for' in query:
            query = query.replace("search google for", "")
            speaker.speak(f"Searching Google for {query}")
            webbrowser.open(f"https://www.google.com/search?q={query}")


        elif 'open notepad' in query:
            speaker.speak("Opening Notepad Sir")
            subprocess.Popen(["notepad.exe"])

        elif 'open file explorer' in query:
            speaker.speak("Opening file explorer Sir")
            subprocess.Popen(["explorer.exe"])

        elif 'open code' in query:
            speaker.speak("Opening Visual Studio Code Sir")
            codePath = "C:\\Users\\DELL\\Desktop\\Gallery\\Application Setup\\VS code\\Microsoft VS Code\\Code.exe"
            os.startfile(codePath)

        elif 'open python' in query:
            speaker.speak("Opening Python IDLE Sir")
            codePath = "C:\\Program Files\\Python311\\Lib\\idlelib\\idle.pyw"
            os.startfile(codePath)

        elif 'open html' in query:
            speaker.speak("Opening notepad++ Sir")
            codePath = 'C:\\Users\\DELL\\OneDrive\\Desktop\\Gallery\\Application Setup\\Notepad++\\notepad++'
            os.startfile(codePath)

        elif 'what time is it' in query:
            current_time = datetime.datetime.now().strftime("%I:%M %p")
            speaker.speak(f"Sir, The current time is {current_time}")

        elif 'wikipedia' in query:
            speaker.speak('Searching Wikipedia...')
            query = query.replace("wikipedia","")
            results = wikipedia.summary(query, sentences=2)
            speaker.speak("According to Wikipedia")
            print(results)
            speaker.speak(results)

        elif 'search ai' in query:
            query = query.replace("search ai","")
            speaker.speak("According to ai :: ")
            speaker.speak(ai(prompt=query))

        elif 'tell me about' in query:
            query = query.replace('tell me about','')

        elif 'today weather condition for ' in query:
            query = query.replace('today weather condition for ','')
            askweather(query)

        elif 'jarvis quit' in query:
            speaker.speak("If you have any more questions or need further assistance in the future, feel free to reach out. Have a great day!")
            break

     
        
