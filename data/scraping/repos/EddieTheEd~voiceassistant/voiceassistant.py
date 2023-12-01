import atexit
import pyttsx3
import sys
import tkinter as tk
from tkinter import ttk
import speech_recognition as sr
from datetime import datetime
import os
import openai

# Startup

global closebyvoice
closebyvoice = False
openai.api_key = None

# clearing the terminal
clear = lambda: os.system('cls')
clear()

# getting the tts set up
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

#finding all account (data to be implemented)
accountdata = []
accountdatatemp = []
accounts = open("accountdata.txt", "r+")
for line in accounts:
    if "\n" in line:
        accountdatatemp.append(line[:-1])
    else:
        accountdatatemp.append(line)

for i in accountdatatemp:
    if "USER:" in i:
        temp = i.split()
        temp.remove(temp[0])
        username = ' '.join(temp)
        accountdata.append(username)

# accountdata is a list containing all usernames, soon to be made into dictionary

#this opens the logging text file
userdata = open('userdata.txt', 'a')

#function to both add lines to log file, and print
def log(message):
    date = str(datetime.now())
    userdata.write(date + "\n" + message + "\n\n")
    print(date + "\n" + message + "\n")

#small function for tts
def speak(audio):
    engine.say(audio)
    engine.runAndWait()


def takeCommand():
    mic = sr.Recognizer()
     
    with sr.Microphone() as source:
         
        boxstatus("Listening...")
        mic.pause_threshold = 1
        audio = mic.listen(source)
  
    try:
        boxstatus('Program recognizing')
        query = mic.recognize_google(audio, language ='en-in')
        boxstatus(f"User said: {query}\n")
  
    except Exception as e: 
        boxstatus('Program failed to understand voice')
        return "None"
     
    return query

def boxstatus(input):
    window.label.destroy()
    window.imageLabel.destroy()
    
    coolmage = tk.PhotoImage(file="./assets/voiceassistlogo.png")
    window.imageLabel = tk.Label(window, image = coolmage)
    
    window.imageLabel.pack()

    window.label = ttk.Label(text = input)
    window.label.pack()
    window.update()
    log(input)

def confirmmanualusername():

    if window.entry.get() == '':
        speak("Please try again.")
        window.entry.destroy()
        window.imageLabel.destroy()
        window.yesbutton.destroy()
        manualtextbox()
    else:
        name = window.entry.get()
        window.imageLabel.destroy()
        window.entry.destroy()
        window.yesbutton.destroy()
        img = tk.PhotoImage(file="./assets/voiceassistlogo.png")
        window.imageLabel = tk.Label(window, image = img)
        
        window.imageLabel.pack()

        window.label = ttk.Label(text = input)
        window.label.pack()

        boxstatus(f'Hello {name}!')
        speak("hello" + name + ", how are you today! i hope you are well")
        accounts.write("USER: " + name + "\n")
        a = True
        while a:
            main()



def manualtextbox():
    global window

    img = tk.PhotoImage(file="./assets/voiceassistlogo.png")
    window.imageLabel = tk.Label(window, image = img)
    
    window.imageLabel.pack()

    window.iconbitmap('./assets/voiceassistlogo.ico')
    window.title("Ed's Voice Assistant!")
    window.geometry('320x130+50+160')

    window.entry = tk.Entry()
    window.entry.pack()
    window.yesbutton = ttk.Button(window, text="Confirm Username", command=confirmmanualusername)
    window.yesbutton.pack()
    window.attributes('-topmost', 1)
    window.update()


def intro():
    speak("what is your name")
    boxstatus('Voice Assistant Status: Listening')
    namecount = 0

    while namecount != 3:
        name = str(takeCommand())
        if name != 'None':
            namecount = 2
            boxstatus(f'Hello {name}!')
            speak("hello" + name + ", how are you today! i hope you are well")
            accounts.write("USER: " + name + "\n")
            a = True
            while a:

                main()
        else:
            boxstatus('Did not understand.')
            if namecount == 0:
                speak("well, i wasn't able to catch your name, could you repeat that?")
            elif namecount == 1:
                speak("weird, i didn't catch that again. please repeat what you said.")
            elif namecount == 2:
                namecount = 3
                log("User failed to register name using voice. Proceeding to text input.")
                speak("Your name must be quite special. Please type it in the box provided.")
                window.label.destroy()
                window.imageLabel.destroy()
                manualtextbox()
                break
               
            namecount += 1

    
            

def finduser(username):
    newdow.destroy()
    global name
    name = username
    maingui()
    boxstatus(f'Hello {name}!')
    speak("hello" + name + ", welcome back.")
    a = True
    while a:

        main()

def accountselection():
    window.destroy()
    global newdow
    newdow = tk.Tk()
    height = int(round(len(accountdata) * (150/6)))
    newdow.geometry(f'320x{height}+470+50')
    newdow.title("Available accounts: ")
    newdow.iconbitmap('./assets/voiceassistlogo.ico')
    for i in accountdata:
        newdow.button = ttk.Button(text = str(i))
        newdow.button['command'] = lambda i=i: finduser(i)
        newdow.button.pack()

def startup():
    boxstatus('Voice Assistant Status: Configuration')
    speak('processing, starting up')
    if not accountdata:
        speak('no previous users detected')
        log('no previous users detected')
        intro()
    elif len(accountdata) == 1:
        global name
        name = accountdata[0]
        log(f'username {name} detected')
        boxstatus(f"Hello {name}!")
        speak("hello" + name + ", welcome back.")
        a = True
        while a:

            main()
    else:
        speak('multiple users detected. please choose your account.')
        accountselection()
        
def main():
    programUnderstood = False
    if openai.api_key == None:

        boxstatus("OpenAI Key required")

        global open
        open = tk.Tk()
        open.geometry('320x70+50+400')
        open.title("Insert OpenAI key:")
        open.iconbitmap('./assets/voiceassistlogo.ico')

        help = tk.Tk()
        help.geometry('320x130+450+160')
        help.title("Commands:")
        help.iconbitmap('./assets/voiceassistlogo.ico')
        help.label = tk.Label(help, text = "Things you can do!\n'obsidian': Opens obsidian\n'Firefox': Opens firefox\n'restart': Restarts\n'close program': Closes program\n'create new account': Creates new account\n'{generic questions}': Will be parsed through\nOpenAI for answer.")
        help.label.pack()
        help.update()
        help.attributes('-topmost', 1)
        boxstatus("OpenAI Key required")

        def askforkey():
            open.label = ttk.Label(open, text = "This is required for the code to work.")
            open.label.pack()
            open.entry = tk.Entry(open)
            open.entry.pack()
            open.yesbutton = ttk.Button(open, text="Confirm Key", command=confirmkey)
            open.yesbutton.pack()
            open.attributes('-topmost', 1)
            open.update()
            boxstatus("OpenAI Key required")

        def confirmkey():
            if open.entry.get() == '':
                open.label.destroy()
                open.entry.destroy()
                open.yesbutton.destroy()
                speak("Try again")
                askforkey()
            else:
                openai.api_key = str(open.entry.get())
                open.destroy()
                main()

        askforkey()
        speak("please input open a i key")
        
        open.mainloop()
        boxstatus("OpenAI Key required")

    speak("what would you like to do?")
    query = str(takeCommand()).split()
    print(query)
    if query[0] != "None":
        boxstatus(f"Analysing input: '{' '.join(query)}'")

    # base functions
        if 'obsidian' in query:
            speak("Understood, opening Obsidian")
            os.startfile("C:\\Users\\edwar\\AppData\\Local\\Obsidian\\Obsidian.exe")
            boxstatus("Success!")
            programUnderstood = True
            main()
        elif 'restart' in query:
            speak("restarting")
            os.execv(sys.executable, ['python'] + sys.argv)
            boxstatus("Success!")
            programUnderstood = True
        elif 'Firefox' in query:
            speak("Understood, opening firefox.")
            os.startfile("C:\\Program Files\\Mozilla Firefox\\firefox.exe")
            boxstatus("Success!")
            programUnderstood = True
            main()
        elif 'close program' in ' '.join(query):
            speak("Understood, see you again soon.")
            log('Program terminated via voice command')
            global closebyvoice
            closebyvoice = True
            exit()
        elif 'create new account' in ' '.join(query):
            speak("all right. I'll forget your current account, and create a new one")
            log("Creating new account")
            intro()

        else:
            # putting it through openai
            speak("putting through open a i")

            response = openai.Completion.create(
            model="text-davinci-002",
            prompt=' '.join(query),
            temperature=0.3,
            max_tokens=150,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            )
            answer = response['choices'][0]['text']
            log("openai request answer: "+ answer)
            speak(str(answer))
            if str(answer) != "None":
                programUnderstood = True
                main()
            

    if programUnderstood == False:
        boxstatus("Failed to understand command.")
        speak("Sorry. i didn't catch that.")
        main()

def maingui():
    global window
    window = tk.Tk()

    img = tk.PhotoImage(file="./assets/voiceassistlogo.png")
    window.imageLabel = tk.Label(image = img)
    
    window.imageLabel.pack()

    window.iconbitmap('./assets/voiceassistlogo.ico')
    window.title("Ed's Voice Assistant!")
    window.geometry('320x130+50+160')

    window.label = ttk.Label(window, text='processing...')
    window.label.pack()

    window.attributes('-topmost', 1)

    window.update()

def exitlog():
    global closebyvoice
    if closebyvoice == False:
        log('Program terminated via script ending.')
    userdata.close()



#initial_window creation, i.e maingui with intialization button
log('Program Initialised')
atexit.register(exitlog)

global window
window = tk.Tk()

#visual stuff
global img
img = tk.PhotoImage(file="./assets/voiceassistlogo.png")

window.imageLabel = tk.Label(window, image = img)
window.imageLabel.pack()

window.iconbitmap('./assets/voiceassistlogo.ico')

#core window code

window.title("Ed's Voice Assistant!")
window.text = tk.StringVar()
window.text.set("Press to start program!")
window.geometry('320x130+50+160')

window.label = ttk.Button(window, textvariable = window.text, command = startup)

window.label.pack()
window.attributes('-topmost', 1)
window.mainloop()