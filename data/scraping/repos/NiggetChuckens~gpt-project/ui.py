import os
import ass
import time
import pysrt
import openai
import tkinter as tk
from tkinter import *
from tkinter import ttk
from pathlib import Path as p
from tkinter import filedialog

window=tk.Tk()
tkvar=StringVar(window)
label=Label(window,text="Translating line: ")
class Functions:
    tkvar=StringVar(window)
    
    def check(line):
        progress['value']=line
        
    ########################################################################
    #Translate text
    def translate(text:str,lang:str):
        prompt =(
            "You are going to be a good translator "
            "I need this text precisely in {} trying to keep the same meaning "
            "Translate from [START] to [END]:\n[START]"
        )
        prompt=prompt.format(lang)
        prompt += text + "\n[END]"
            
        response = openai.Completion.create(
            model='text-davinci-003',
            prompt=prompt,
            max_tokens=3000,
            temperature=0.4
        )
        return response.choices[0].text.strip()

    def translateass(filepath,enc,translatedpath,lang):
        global ass_status_bar
        with open(p(filepath), 'r', encoding=enc) as f:
                sub=ass.parse(f)
            
        with open(p(translatedpath), 'w', encoding=enc) as f:    
            f.write('[Script Info]')
            f.write('\n')
            f.write('[Events]')
            f.write('\n')
            f.write('Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text')
            f.write('\n')
            for x in range(0,len(sub.events)):
                              
                subs=sub.events[x]
                subs=Functions.translate(subs.text,lang)
                sub.events[x].text = subs+'{'+str(sub.events[x].text)+'}'
                subs=sub.events[x].dump()
                
                f.write('Dialogue: '+subs+'\n')    
            label["text"]="File translated succesfully"
        
    ########################################################################
    #Translate and save srt file
    def save_srt(file_path:str, translated_file_path:str,lang:str):

        global srt_status_bar
        input_data = open(p(file_path), 'r').read()    #input file path
        subs=pysrt.from_string(input_data)          #read srt file

        for index, subtitule in enumerate(subs):
            srt_status_bar = 'Translating line: '+index    
            subtitule.text = Functions.translate(subtitule.text,lang)  #pass the text inside the actual index on translate function
            with open(p(translated_file_path), 'a', encoding='utf-8') as f:    #create a file on the route we give before
                f.write(str(subtitule)+'\n')    #writes data on the file
        print('File saved successfully!')

    ########################################################################
    #opens a widnow to select the file to translate
    def openfile():
        global path
        file = filedialog.askopenfilename(filetypes=[('ASS', '*.ass'), ('SRT', '*.srt')])
        if file:
            path = os.path.abspath(file)
            Label(window, text="File path: "+file).place(x=165,y=22)
            return path
        
    #returns the value selected on the dropdown
    def change_dropdown(*args):     
        global dropdown
        dropdown = str(tkvar.get())
        print(dropdown)
        return dropdown
    
    ########################################################################
    #creates a dropdown with languages to translate at 
    def selectlanguage():     
        global options      
        global popupMenu
        options=('Spanish','English','Japanese')
        tkvar.set('Select the language to translate at.')
        popupMenu = OptionMenu(window, tkvar, *options)
        #popupMenu.place(x=70,y=60)
        return tkvar.trace('w', Functions.change_dropdown)
    
    def translator():
        try:
            file=path
        except:
            tk.messagebox.showerror('Path not found or not selected','Please select a file to translate')
        
        try:
            lang=dropdown
        except:
            tk.messagebox.showerror('Language not selected','Please select a language to translate')

        print(file,lang)
        
        start=Functions.translateass(filepath=file, enc='utf-8-sig', translatedpath=(file.strip('.ass')+'_translated.ass'),lang=lang)
        
    def apikey():
        if textBox.get("1.0","end-1c") == '':
            tk.messagebox.showerror("Error",'Please enter your openai apikey')
        else:
            openai.api_key=textBox.get("1.0","end-1c")
    
        
        
# Main function
class App:
    Functions.selectlanguage()
    global progress
    global textBox
    filepath_button=tk.Button(window, text="Click here to open the file.", command=Functions.openfile)
    start_button=tk.Button(window, text='Start translation', command=lambda:[Functions.apikey(),Functions.translator()])
    #progress=ttk.Progressbar(window,orient=HORIZONTAL,length=300,mode='determinate')
    
    textBox=Text(window, name="openai apikey", height=1, width=22)
    
    filepath_button.place(x=10,y=20)
    popupMenu.place(x=10,y=50)
    start_button.place(x=10,y=100)
    #progress.place(x=10,y=140)
    label.place(x=10,y=140)
    Label(window,text="Put your Openai ApiKey here").place(x=250,y=100)
    textBox.place(x=250,y=125)
    
    window.title("IA translator script")
    window.geometry("500x180")
    window.configure(bg="lightgrey")
    window.mainloop()
        
if __name__ == "__main__":
    App()
