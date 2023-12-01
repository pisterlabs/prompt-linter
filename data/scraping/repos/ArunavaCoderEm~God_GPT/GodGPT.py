# Import Module

import openai
import pyttsx3
import tkinter as tk
from tkinter import *
from tkinter import messagebox
import speech_recognition
import pyttsx3
import requests
import json

# splash screen

splash = Tk()
splash.title ('GodardGPT')
splash.geometry ('400x400')
splash.resizable (0,0)
splash.configure (background = 'red')
spl_lab = Label (splash, text = 'GodARD Devos \n GodardGPT', bg = 'yellow', fg = 'blue', font = ('vendana', 20, 'bold'))
spl_lab.pack (pady = 150)

# # Initialize The Introduction Function

def say():
    global recognizer
    recognizer = speech_recognition.Recognizer()
    engine = pyttsx3.init()
    engine.setProperty("rate",150)
    engine.say(f"Hey I am God here")
    engine.runAndWait()  

# Initialize main window

def main_window_GPT():
        splash.destroy()
        gptwindow = tk.Tk()
        gptwindow.geometry ('1200x500')
        gptwindow.title ("GodardGPT")
        gptwindow.configure (background = '#BF3EFF')
        gptwindow.resizable (0,0)
        gptwindow.minsize (width = 950, height = 600)
        say()
                
        def clearbtn():
            query.delete(0,END)
            qstan.destroy()
            answers.destroy()
            
        def askspeakbtn():
            query.delete(0,END)
            recognizer = speech_recognition.Recognizer()
            engine = pyttsx3.init()
            engine.say(f"Listening")
            engine.runAndWait()
            with speech_recognition.Microphone() as mic:
                recognizer.adjust_for_ambient_noise(mic,duration=0.3)
                audio = recognizer.listen(mic)
                text = recognizer.recognize_google(audio)
                text = text.lower()
                query.insert (0,text) 
        
        def enterbtn():
            global recognizer
            global qstan
            global answeroq
            global question
            global answertq  
            global answers      
            question = str(query.get())
            #
            prompt = (f'please give me the answer of {question}')
            openai.api_key = 'YOUR OWN API KEY THAT YOU WILL GET FROM OPENAI WEBSITE'
            response = openai.ChatCompletion.create(
                model = 'gpt-3.5-turbo',
                messages=[
                    {"role":"user","content": prompt}
                ]
            )
            answertq = response.choices[0].message.content
            #
            
            if (len(question) == 0 or question.isspace() == True):
                qstan = Label (gptwindow, bg = 'white',fg='black',text=f'Please Give Me Some Question ',font=('verdana',10,'bold'))
                qstan.place(x=20,y=350) 
                recognizer = speech_recognition.Recognizer()
                engine = pyttsx3.init()
                engine.setProperty("rate",150)
                engine.say(f"Please Give Me Some Question")
                engine.runAndWait()  
            else :
                qstan = Label (gptwindow, bg = 'white',fg='black',text=f'Your Answer To The Question "{question}" is - ',font=('verdana',10,'bold'))
                qstan.place(x=20,y=350)
                recognizer = speech_recognition.Recognizer()
                engine = pyttsx3.init()
                engine.setProperty("rate",150)
                engine.say(f"Your Answer To The Question {question} is")
                engine.runAndWait() 
                answeroq = str(answertq)         
                answers = Label (gptwindow,bg='black',fg='white',font=('verdana',10,'bold'),text=f'Answer : \n  {answeroq}')
                answers.place(x = 20, y = 380)
                            
        def readaloud():
            global recognizer
            if (len(answeroq) == 0 or answeroq.isspace() == True):
                recognizer = speech_recognition.Recognizer()
                engine = pyttsx3.init()
                engine.setProperty("rate",150)
                engine.say(f"No Answer Detected")
                engine.runAndWait()             
            else :
                recognizer = speech_recognition.Recognizer()
                engine = pyttsx3.init()
                engine.setProperty("rate",150)
                engine.say(f"{answeroq}")
                engine.runAndWait() 

        def quitout ():
            messagebox.showinfo ( 'GPTwindow', 'You Want To Exit ? \n Click "OK" If You Want To !! ')
            return gptwindow.destroy()
        
        #Frames
        
        frameh = Frame(gptwindow,width=1200,height=80,bg='blue')
        frameh.place(y = 20)
        
        frameb = Frame(gptwindow,bg='black',width=1200,height=290)
        frameb.pack(side='bottom')
                
        #Labels 
        
        labelh = Label(frameh,text='Enter Or Ask Your Query Here',bg='blue',fg='yellow',font=('vendana',40,'bold'))
        labelh.place(x = 200)
        
        labelj = Label(frameh,text='Press Read Aloud To Hear Answer',bg='blue',fg='yellow',font=('vendana',10,'bold'))
        labelj.place(x = 500, y = 60)
        
        labelr = Label (gptwindow,text='Result Outcome',bg = 'yellow',fg='blue',font=('vendana', 20,'italic'))
        labelr.place(x = 480, y= 250)
        
        #Entry
        
        query = Entry (gptwindow, width = 45, bg = 'pink', fg = 'red', borderwidth = 4,
        font = ('vendana', 20,'italic'))
        query.insert (0,"Speak or enter query here ...")
        query.place(x= 290, y = 130)
        
        #Buttons
        
        clear = Button (gptwindow, width=10, height=1,text='Clear',bg='brown', fg ='pink',command=clearbtn,font=('vendana',10,'bold'))
        clear.place(x = 300, y = 190)
        
        askspeak = Button (gptwindow, width=10,height=1,text='Speak',bg ='brown', fg='pink',command=askspeakbtn,font=('vendana',10,'bold'))
        askspeak.place(x = 450, y = 190)
        
        enter = Button (gptwindow,width=10,height=1,font=('vendana',10,'bold'),bg='brown',fg = 'pink',text='Enter',command=enterbtn)
        enter.place(x = 600, y =190)
        
        readaloud = Button (gptwindow,width=10,height=1,bg ='brown', fg='pink',font=('vendana',10,'bold'),command=readaloud,text='Read Aloud')
        readaloud.place(x = 750, y = 190)
        
        quit = Button(gptwindow,width=10,height=1,bg ='brown', fg='pink',font=('vendana',10,'bold'),text='Quit',command=quitout)
        quit.place(x = 880,y=190)
        
# spalsh screen timer

splash.after ( 2500, main_window_GPT)

# run app

mainloop()
