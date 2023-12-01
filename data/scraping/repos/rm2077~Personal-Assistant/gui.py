import tkinter as tk
from tkinter import *
import speech_recognition as sr
import openai
import pyttsx3

openai.api_key = "YOUR_API_KEY"
ending_statements = ["break", "stop", "quit", "exit", "bye", "goodbye", "end", "finish", "done", "terminate", "kill", "halt", "cease", "conclude", "close", "wrap up", "shut down", "abort", "cancel", "complete", "conclude", "culminate", "finalize", "finish", "halt", "stop", "terminate", "wind up", "wrap up", "break off", "break up", "cease", "close down", "close out", "close up", "come to an end", "conclude", "desist", "determine", "discontinue", "end", "finish", "give over", "halt", "leave off", "let up", "pack in", "pause", "quit", "refrain", "relax", "relinquish", "rest", "restore", "result", "run out", "say goodbye to", "scrub", "shut down", "sign off", "stay", "stop", "suspend", "terminate", "wind down", "wrap up"]


def speech_to_text():
  r = sr.Recognizer()
  with sr.Microphone() as source:
      print("Say something!")
      audio = r.listen(source)
      

  prompt = r.recognize_google(audio)
  return prompt


def get_openai_response(prompt):
  response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=f"Your name is Voxi. You are a helpful virtual assistant. You are an AI and you have to respond to this: {prompt}",
    temperature=0.9,
    max_tokens=150,
    top_p=1,
    frequency_penalty=0.0,
    presence_penalty=0.6,

  )
   
  txt = response["choices"][0]["text"]
  return txt

def text_to_speech(txt):
  engine = pyttsx3.init()
  voices = engine.getProperty('voices')
  engine.setProperty('voice', voices[1].id)
  engine.say(txt)
  engine.runAndWait()
  print("Virtual assistant: ", txt)


def main():
  while True:
    prompt = speech_to_text()
    if prompt in ending_statements:
      break
    else:
      txt = get_openai_response(prompt)
      text_to_speech(txt)


root = tk.Tk()
root.title("Your personal virtual assistant")
root.geometry("600x500")
root.configure(bg="black")
label = Label(root, text="Alissa, your virtual assistant", font=("Arial", 20, "bold"), fg="white", bg="black")
label.pack()
button = Button(root, text="Start your conversation", font=("Arial", 15), fg="white", bg="black", command=main)
button.pack()
root.mainloop()
