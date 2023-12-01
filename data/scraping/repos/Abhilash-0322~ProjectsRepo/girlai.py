import speech_recognition as sr
import win32com.client
import webbrowser
import datetime
import openai
speaker=win32com.client.Dispatch("SAPI.SpVoice")
openai.api_key = "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"
def gptfunction(query):
# user_input = input("Enter your input: ")
    

    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "user",
            "content": "act as me"
        },
        # {
        #     "role": "user",
        #     "content": "You mean so much to me, and I really appreciate your caring nature."
        # },
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
    with open ('girlai.txt','a') as f:
        g=generated_text
        f.writelines(g)
        f.writelines('\n')
    return generated_text
def takeCommand():
    r=sr.Recognizer()
    with sr.Microphone() as source:
        r.pause_threshold=0.6
        audio=r.listen(source)
        try:
            print('recognizing')
            query=r.recognize_google(audio,language='en-us'and'en-in')
            print(f"User said:{query}")
            return query
        except Exception as e:
            print("What Do You Mean, say again")
            return "What Do You Mean, say again"

while True:
    print("listening....")
    query=takeCommand()
    responses=gptfunction(query)
    #to add more websites
    sites=[["youtube","https//youtube.com"],["wikipedia","https://www.wikipedia.org"],["gmail","https://www.gmail.com"], ["github", "https://www.github.com"]]
    for site in sites:
        if f"open {site[0]}" in query.lower():
            webbrowser.open(f"{site[1]}")
            print(f"opening {site[0]}")
    
    #to add tasks
    if "the time" in query:
        strfTime=datetime.datetime.now().strftime("%H:%M:%S")
        print(strfTime)
        speaker.speak(f"Time is {strfTime}")
    if "weather" in query:
        speaker.Speak("I'm sorry, I can't provide weather information at the moment.")  
    print(f"AI response: {responses}")
    speaker.Speak(responses)  