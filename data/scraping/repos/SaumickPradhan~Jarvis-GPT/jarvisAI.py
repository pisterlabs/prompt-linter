import speech_recognition as sr
import openai
import os
import webbrowser
import datetime
from config import OPENAI_API_KEY


def ai(prompt):
    openai.api_key = OPENAI_API_KEY
    text = f"OpenAI response for Prompt: {prompt} \n *************************\n\n"
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # todo: Wrap this inside of a try-except block
        text += response["choices"][0]["text"]
        print(text)
    except Exception as e:
        print(f"Some Error occurred: {e}")

    # if not os.path.exists("Openai"):
    #     os.mkdir("Openai")

    # with open(f"Openai/{''.join(prompt.split('intelligence')[1:]).strip()}.txt", "w") as f:
    #     f.write(text)


def say(text):
    os.system(f"say {text}")


def takeCommand():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.energy_threshold = 50
        #recognizer.pause_threshold = 1
        #recognizer.adjust_for_ambient_noise(source)
        #recognizer.dynamic_energy_threshold = False
        audio = recognizer.listen(source)
        try:
            print("Recognizing...")
            query = recognizer.recognize_google(audio, language='en-US')
            print(f"User said: {query}")
            return query
        except Exception as e:  
            print(e)
            return "Some Error Occurred."
    

if __name__ == '__main__':
    say("Mac Assistant is ready!")
    while True:
        print("Listening...")
        query = takeCommand()
        websites = [["youtube", "https://www.youtube.com"], ["wikipedia", "https://www.wikipedia.com"], ["google", "https://www.google.com"]]
        for site in websites:
            if f"Open {site[0]}".lower() in query.lower():
                say(f"Opening {site[0]}...")
                webbrowser.open(site[1])
        if "the time".lower() in query.lower():
            time = datetime.datetime.now().strftime("%H:%M")
            say(f"Sir time is {time}")
        
        elif "open facetime".lower() in query.lower():
            os.system(f"open /System/Applications/FaceTime.app")

        elif "Using artificial intelligence".lower() in query.lower():
            ai(query)

        elif "Jarvis Quit".lower() in query.lower():
            exit()

        else:
            print("Chatting...")
            #chat(query)


#    elif "reset chat".lower() in query.lower():
#             chatStr = ""


# def chat(query):
#     global chatStr
#     print(chatStr)
#     openai.api_key = OPENAI_API_KEY
#     chatStr += f"Harry: {query}\n Jarvis: "
#     response = openai.Completion.create(
#         model="text-davinci-003",
#         prompt=chatStr,
#         temperature=0.7,
#         max_tokens=256,
#         top_p=1,
#         frequency_penalty=0,
#         presence_penalty=0
#     )
#     # todo: Wrap this inside of a try-except block
#     say(response["choices"][0]["text"])
#     chatStr += f"{response['choices'][0]['text']}\n"
#     return response["choices"][0]["text"]







