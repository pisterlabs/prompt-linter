import datetime
from logging import exception
import webbrowser
import speech_recognition as sr
import pyttsx3
import wikipedia
import os
import smtplib,ssl
import pywhatkit as kit
import playsound
import openai
from gtts import gTTS

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
#print(voices[1].id)
engine.setProperty('voice',voices[1].id)

api_key="enter your api key"       #api is used here 
openai.api_key = api_key



def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def wishme():                                   #for wishyou good morning ,evening and night
    hour=int(datetime.datetime.now().hour)
    if hour>=0 and hour<12:
        speak('good morning!')
    elif hour>=12 and hour<18:   
        speak('good afternoon!')
    else:
        speak('good evening!')

    speak(' i am maven sir. please tell me how may i help you') 

def  takecommand():                                # for taking commands
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("listening.....")
        r.pause_thresold = 1
        audio = r.listen(source)
    
    try:
        print('recongizing.....')    
        query= r.recognize_google(audio , language="en-in")
        print(f'user-said: {query}\n')
        
        if "Maven" in query:                                                       #code to get answer for gpt and first by saying maven it recognize then its stats this query;
            speak("Searching text....")
            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=[{"role": "user", "content": query}])
            text=completion.choices[0].message.content
            speech= gTTS(text=text, lang="en-in", slow=False, tld="com.au" )
            # speech.save("welcome.mp3")
            # playsound.playsound("welcome.mp3")
            print(text)
            speak(speech)
    
    except Exception as e:
        #print(e)
        
        print('Sir will you Say that again.....')
        speak('Sir will you Say that again.....')    # if not recognize 
        return 'none'   
    return query    
        
def sendEmail(to,content):
    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"
    sender_email = "gmail.com"  # Enter your address
    
    password = input("Type your password and press enter: ")
    
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, to, content)
    
def message(voicemess):    
    kit.sendwhatmsg_instantly("+91 79XXXXXXXX",voicemess,15)   # for whatsapp msssg

# def timer(hour,minute,voicemess):
#     kit.sendwhatmsg("+91 79XXXXXXXX",voicemess,hour,minute)
        

if __name__=="__main__":
    wishme()
    while True:
     if 1:
   
        query=takecommand().lower()
        
        #logic for executing task based on query
        if "wikipedia" in query:
            speak("Searching wikipedia....")
            query=query.replace('wikipedia..',"")
            results= wikipedia.summary(query, sentences=2)
            speak("according to wikipedia...")
            print(results)
            speak(results)
        
        elif "youtube " in query:
            webbrowser.open("youtube.com")
        
        elif "open google " in query:
            webbrowser.open("google.com")
        
        elif "the time " in query:
            stime=datetime.datetime.now().strftime("%H:%M:%S")
            speak(f"Sir, the time is {stime}")
        
        elif "open vs code  " in query:
            vs="C:\\Users\\Sanket\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe" 
            os.startfile(vs)
        
        elif "send email " in query:
            try:
                speak("what should i send")
                content= takecommand()
                to = "jaiswar00@gmail.com"
                sendEmail(to,content)
                speak("email has sent")
            except Exception as e: 
                print(e)
                speak("sorry email not send") 
         
        elif "send message" in query:
            try:
                speak("what is your message, sir ......")
                voicemess = takecommand()
                message(voicemess)
                # speak("what type first is isntant or timer")
                # tyype=takecommand()
                # first = 1
                # second = 2    
                # if tyype==first:
                #     message(voicemess)
                # elif tyype==second:
                #     speak("what the hour")
                #     hour =takecommand()
                #     speak("and minute")
                #     minute =takecommand()
                #     timer(hour,minute,voicemess)                       
                                    
            except Exception as e:   
                print(e)
                speak("sorry to send mess") 
                    
        elif "quit" in query:      #exit the code
            exit()