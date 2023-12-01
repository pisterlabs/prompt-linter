from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from pydub import AudioSegment
from tkinter import scrolledtext
from openai import OpenAI
import os

client = OpenAI(api_key="Insert Here your openAI key")
root = Tk()
frm = ttk.Frame(root, padding=2)
frm.grid()
def browseFiles():
    filename = filedialog.askopenfilename(initialdir = "/",
                                          title = "Select a File",
                                          filetypes = (("all files","*.*"),("OGG files","*.ogg")))
    AudioSegment.from_file(filename).export("./temp.mp3", format="mp3")
    audio_file= open("./temp.mp3", "rb")
    transcript = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file,
        response_format="text"
    )
    txt.insert(INSERT, transcript)
    os.remove("./temp.mp3")

def clear(): 
    txt.delete('1.0','end')
    

ttk.Label(frm, text="Hello World!").grid(column=1, row=0)
ttk.Button(frm, text="Quit", command=root.destroy).grid(column=1, row=0)
ttk.Button(frm,text="Clear",command= clear).grid(column=1, row=2 )
txt = scrolledtext.ScrolledText(root, width=100, height=10)
txt.grid(column=1, row=0)
button_explore = ttk.Button(frm, 
                        text = "Browse Files",
                        command = browseFiles).grid(column = 1, row = 1)
root.mainloop()

