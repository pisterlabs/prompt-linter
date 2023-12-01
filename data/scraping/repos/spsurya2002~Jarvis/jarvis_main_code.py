import webbrowser
import speech_recognition as sr
import os
import win32com.client
import datetime
import openai
from config import apikey

speaker = win32com.client.Dispatch("SAPI.Spvoice")

def say(text):
    speaker.Speak(text)

chatStr = ""
# https://youtu.be/Z3ZAJoi4x6Q
def chat(query):
    global chatStr
    print(chatStr)
    openai.api_key = apikey

    chatStr += f"Harry: {query}\n Jarvis: "
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt= chatStr,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    # todo: Wrap this inside of a  try catch block
    say(response["choices"][0]["text"])
    chatStr += f"{response['choices'][0]['text']}\n"
    return response["choices"][0]["text"]


def ai(prompt):
    openai.api_key = apikey
    text = f"OpenAI response for Prompt: {prompt} \n *************************\n\n"

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    # todo: Wrap this inside of a  try catch block
    # print(response["choices"][0]["text"])
    text += response["choices"][0]["text"]
    if not os.path.exists("Openai"):
        os.mkdir("Openai")

    # with open(f"Openai/prompt- {random.randint(1, 2343434356)}", "w") as f:
    with open(f"Openai/{''.join(prompt.split('intelligence')[1:]).strip() }.txt", "w") as f:
        f.write(text)








def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.pause_threshold = 0.8
        audio = r.listen(source)
        try:
            query = r.recognize_google(audio, language='en-in')
            print(f"User said: {query}")
            return query
        except Exception as e:

            return "sorry ! command can't be recognized"

if __name__ == '__main__':

    say("hello! I am jarvis! Welcome to my world")
    # Get current date and time
    now = datetime.datetime.now()
    # Get the day, month, year, and time
    day = now.strftime("%A")
    month = now.strftime("%B")
    year = now.year
    hour = now.strftime("%H")
    min = now.strftime("%M")
    # saying current timestamp
    # say("Today is"+ str(day)+ "of"+ str(month)+ str(year))
    # say("The current time is"+str(int(hour))+"bajke"+str(int(min))+"minute")

    while 1:
        print("Listenning...")
        say("listenning...")
        query = takeCommand()

        sites=[["youtube","https://www.youtube.com/"],
               ["leetcode","https://leetcode.com/studyplan/top-interview-150/"],
               ["codeforces","https://codeforces.com/profile/Surya0123"],
               ["udemy","https://www.udemy.com/home/my-courses/learning/"]
               ]
        for site in sites:
            if f"{site[0]}".lower() in query.lower():
                say(f"openning {site[0]} sir...")
                webbrowser.open(site[1])
                break
        if query.lower()=="jarvis shutup"or query.lower()=="jarvis quit"or query.lower()=="shut up" :
            say("i am shutting down!")
            break
        elif query.lower() == "music":
            music_path = 'C:\\Users\\HP\\Desktop\\Project(backend)\\jarvisAI\\Memories---Maroon-5-320-(PagalWorld).mp3'
            say("enjoy!! music is playing...")
            os.startfile(music_path)
            break
        elif "Using artificial intelligence".lower() in query.lower():
            ai(prompt=query)
        elif "reset chat".lower() in query.lower():
            chatStr = ""
        else:
            print("Chatting...")
            chat(query)
