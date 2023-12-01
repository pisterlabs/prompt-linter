import pyttsx3
import datetime as dt
import wikipedia as wiki
import webbrowser as wb
from tkinter import *
from tkinter import ttk
import calendar
from num2words import num2words
import whisper
import sounddevice as sd
import soundfile as sf
import requests
import openai

fref = open('Desktop/Jarvis/keys.txt','r', newline = '\n')
key_string = fref.read()
fref.close()
keys = key_string.split("_")

OPENAI_Key = keys[0]
News_Key = keys[1]
Movie_Key = keys[2]
Ow_Key = keys[3][:-1]


base_url = "http://api.openweathermap.org/data/2.5/weather?"
openai.api_key = OPENAI_Key


model = whisper.load_model("small")

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[7].id)



def speak(text):
    engine.say(text)
    engine.runAndWait()

def hello():
    infoStr = str(dt.datetime.now())
    min = infoStr[14:16]
    hour = infoStr[11:13]
    day = num2words(infoStr[8:10], "ordinal_num")
    weekday = day_of_week(dt.datetime.now().weekday())
    month = calendar.month_name[int(infoStr[5:7])]
    greeting = ""
    ampm = ""
    oh = ""
    if -1 < int(hour) < 13:
        ampm = "AM"
        greeting = "good morning"
    elif 12 < int(hour) < 16:
        ampm = "PM"
        greeting = "good afternoon"
        hour = str(int(hour) - 12)
    elif 15 < int(hour) < 25:
        ampm = "PM"
        greeting = "good evening"
        hour = str(int(hour) - 12)

    if int(min) < 10:
        oh = "oh"

    speak(greeting + " sir, today is " + weekday + month + day + ", and the current time is " + hour + " " + oh + min + ampm)

def listen():
    fs = 48000
    duration = 5
    myrecording = sd.rec(int(duration*fs), samplerate=fs, channels=1)
    sd.wait()
    sf.write("my_audio_file.flac", myrecording, fs)
    audio = "my_audio_file.flac"
    options = {"fp16" : False, "language" : "English", "task" : "transcribe"}
    results = model.transcribe(audio, **options)
    query = results['text'].lower()
    while query[0] == " ":
        query = query[1:]
    print("Query: \'" + query + "\'")
    process_audio(query)

def process_audio(query):
    if "open" in query:
        site = website(query)
        speak("Opening " + site + "now sir")
        site = site.replace(' ', '')
        wb.open_new("https://www." + site + ".com")

    elif "what is the weather" in query:
        city = get_city(query)
        result = weather(city)
        if result["error"] == 0:
            speak("The current temperature in " + city + " is " + result["temp"] + " degrees fahrenheit with " + result["desc"])
        else:
            speak("Sorry, city not found")

    elif "from wikipedia" in query:
        question = query.replace("from wikipedia", "")
        result = wiki.summary(question, sentences=2)
        speak(result)

    elif "popular movies" in query:
        movies = popular_movies()
        speak("Here are the top 5 most popular movies out now: ")
        speak(movies)

    elif "today's news" in query:
        headlines = news()
        speak("here are the top headlines for today: ")
        print(headlines)
        speak(headlines)

    else:
        speak("One moment please")
        answer_raw = openai.Completion.create(
            model="text-davinci-003",
            prompt=query,
            max_tokens=10,
            temperature=0.2,
            )
        answer = answer_raw.choices[0].text[2:]
        speak(answer)

    speak("Goodbye sir")

def get_city(query):
    city = query.replace("what is the weather in", "")
    city = city.replace("?","")
    city = city.replace(".","")
    return city

def day_of_week(num):
    if num == 0:
        return "Monday"
    if num == 1:
        return "Tuesday"
    if num == 2:
        return "Wednesday"
    if num == 3:
        return "Thursday"
    if num == 4:
        return "Friday"
    if num == 5:
        return "Saturday"
    if num == 6:
        return "Sunday"

def weather(city):
    url = base_url + "appid=" + OW_Key + "&q=" + city
    response = requests.get(url)
    info = response.json()
    print(info)
    if info["cod"] == '404':
        return {"error" : 1, "temp" : 0, "desc" : 0}
    else:
        mainInfo = info["main"]
        temp = mainInfo["temp"]
        weather = info["weather"]
        weather_description = weather[0]["description"]
        temp = str(round(1.8*(temp-273)+32))
        desc = str(weather_description)
        error = 0
        return {"error" : error, "temp" : temp, "desc" : desc}

def website(query):
    site = query.replace("open", '')
    site = site.replace(".","")
    return site

def popular_movies():
    movies = []
    info = requests.get(f"https://api.themoviedb.org/3/trending/movie/day?api_key=" + Movie_Key).json()
    results = info["results"]
    for movie in results:
        title = movie["original_title"]
        movies.append(title)
    response = (', '.join(movies[:4])) + ", and " + movies [4]
    return response

def news():
    headlines = []
    info = requests.get(f"https://newsapi.org/v2/top-headlines?country=us&apiKey=" + News_Key + "&category=general").json()
    articles = info["articles"]
    for article in articles:
        title = article["title"]
        headlines.append(title)
    response = ', '.join(headlines[:5])
    response.replace("-", ",")
    return response


def info():
    info_screen = Toplevel(screen)
    info_screen.geometry("250x150")
    info_screen.title("Info")

    creator_label = Label(info_screen, text = "Created by Patrick Lyons")
    creator_label.pack()

    job_label = Label(info_screen, text = "Student at Cornell University")
    job_label.pack()

    email_label = Label(info_screen, text = "patlyons252@gmail.com")
    email_label.pack()

# running code
hello()

screen = Tk()
screen.title("Jarvis")
screen.geometry("325x250")

ttk.Button(screen, text ="Listen", command = listen).pack()
ttk.Button(screen, text ="Info", command = info).pack()

Label(screen, text = "Click \"Listen\" above when ready to ask question. \n Ask any question or choose from prompts below: \n \n \"What are popular movies out now?\" \n \"What's today's news?\" \n \"Open <website>\" \n \"Tell me a fun fact\" \n \"What is the weather in <city>\" \n \"From Wikipedia, <question>\"", font = "Calibri").place(x=5,y=70)

screen.mainloop()
