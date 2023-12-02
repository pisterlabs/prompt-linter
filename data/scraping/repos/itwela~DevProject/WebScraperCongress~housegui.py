# This is a GUI for a web scraper that pulls PTR (Periodic Trading Records) from whoever I choose.
import tkinter as tk
from tkinter import ttk 
from tkinter import messagebox
from tkinter import font
import subprocess
import os
import customtkinter
import downloadFile2020 as df20
import downloadFile2021 as df21
import downloadFile2022 as df22
import downloadFile2023 as df23
import house as ho
import pandas as pd
import fileinput
import openai

# OpenAI API key setup start----
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    raise Exception("API key not found in .env file")

openai.api_key = api_key
# OpenAI API key setup finish----


#customtkinter.set_appearance_mode('Dark')
customtkinter.set_default_color_theme('blue')


# window
app = customtkinter.CTk()
app.geometry('720x580')
app.title('CONGRESS WEB SCRAPER')

#
cfont = customtkinter.CTkFont(family='Arial', size=16)
gptfont = customtkinter.CTkFont(family='Arial', size=12)
tfont = customtkinter.CTkFont(family='Arial', size=24, weight='bold')

# funciton for year input (excited as i figured this out myself :)
def startdownload():
    try:
        year = year_input.get()
        filename = "downloadFile{}.py".format(year)
        subprocess.call(["python", filename])
        #success
        finishLabel.configure(text='Your files are ready!', text_color='#90ee90')
    except:
        #error
        finishLabel.configure(text='There has been an Error!', text_color='red')
    


#function for searching the ptr (house)
def houseparse():
    try:
        year = year_input.get()
        lastname = last_name_var.get()
        textfilename = "{}FD.txt".format(year)
        subprocess.call(["python", 'house.py', textfilename, lastname])

        # Get a list of all .txt files in the current directory
        txt_files = [file for file in os.listdir() if file == "politician_filings.txt"]

        # Sort the files by their creation time
        txt_files.sort(key=lambda x: os.path.getctime(x), reverse=True)


        if txt_files:
            most_recent_txt_file = txt_files[0]
            with open(most_recent_txt_file, "r") as file:
                file_contents = file.read()

            # Use OpenAI GPT-3 API to generate text based on file_contents
            # prompt = f"Give me the name, filing date and docids in a table view please. Dont do any kind of explinations just give me exactly what im asking for thank you. Put it in the format like this, Name: ,FilingDate:, DocID:. After that, give me links that i can put into google in this format. You will be replacing the year and docid only. heres an example of the link: https://disclosures-clerk.house.gov/public_disc/financial-pdfs/2020/10038978.pdf. in this link '2020' is the year and 10038978 is the DocId. {file_contents}"
            prompt = f"give me links that i can put into google in this format. You will be replacing the StateDst and docid only. heres an example of the link: https://disclosures-clerk.house.gov/public_disc/financial-pdfs/2020/10038978.pdf. in this link '2020' is the StateDst and 10038978 is the DocId. dont give me anything else other than the links and you can put the name and filing date before you provide me each link just to help organize. Dont explain anything. Just follow instructions. Just a list of links paired with the name and filing date.  {file_contents}"
            response = openai.ChatCompletion.create (
                model="gpt-4",
                temperature=0,
                messages=[
                    {
                "role": "system",
                "content": "You are a highly skilled AI trained in language comprehension and summarization. keep it short and keep the answer far under 8000 tokens and most importantly make following instructions your number one priority."
                    },
                    {
                "role": "user",
                "content": prompt
                    }
                ],
                max_tokens=1000  # Adjust the max_tokens as needed
             )

            generated_summary = response['choices'][0]['message']['content']

            print(generated_summary)
            #success
            finishLabel.configure(text='Success!', text_color='#90ee90')
            output_label.insert(index="0.0", text=generated_summary, tags=None)
    except subprocess.CalledProcessError as e:
        print("Error:", e)
        # Handle the error
    except Exception as e:
        print("An error occurred:", str(e))
        # Handle other exceptions


#title
title = customtkinter.CTkLabel(app, text='CONGRESS PTR WEB SCRAPER', font=tfont)
title.pack(padx=10, pady=25)

#year text
year_title = customtkinter.CTkLabel(app, text='Insert a year', font=cfont)
year_title.pack(padx=10, pady=10)

#input for the year
year_var = tk.StringVar()
year_input = customtkinter.CTkEntry(app, width=300, height=40, textvariable=year_var)
year_input.pack()

# download button
download = customtkinter.CTkButton(app, text="Download", command=startdownload)
download.pack(padx=10, pady=10)

#year text
last_name_title = customtkinter.CTkLabel(app, text='Insert a last name', font=cfont)
last_name_title.pack(padx=10, pady=10)

#input for the last name
last_name_var = tk.StringVar()
last_name_input = customtkinter.CTkEntry(app, width=300, height=40, textvariable=last_name_var)
last_name_input.pack()

# find button
find = customtkinter.CTkButton(app, text="Find", command=houseparse)
find.pack(padx=10, pady=10)


# Finished Downloading
finishLabel = customtkinter.CTkLabel(app, text='')
finishLabel.pack()

#custom data
output_label = customtkinter.CTkTextbox(app, font=gptfont, width=500, height=150)
output_label.place(relx=0.5, rely=0.84, anchor="center")
# instructions
usage_label_1 = customtkinter.CTkLabel(app, text='Step 1. Enter a year of your choosing and press Download.', wraplength=100)
usage_label_1.place(relx=0.20, rely=0.18, anchor="ne")

usage_label_2 = customtkinter.CTkLabel(app, text='Step 2. Enter a Last Name of a Politician you are curious about. Ex: Banks, Thanedar and the press Find.', wraplength=100)
usage_label_2.place(relx=0.20, rely=0.40, anchor="ne")

steps = customtkinter.CTkLabel(app, text='Once you get your results, just copy which ever link fits your fancy and search away!', wraplength=100)
steps.place(relx=0.80, rely=0.18, anchor="nw")

ex_1 = customtkinter.CTkLabel(app, text='Some Popular Politicians include Steny Hoyer, Robert Aderholt, Shri Thanedar, Nancy Pelosi, Suzan K DelBene, etc.', wraplength=100)
ex_1.place(relx=0.80, rely=0.40, anchor="nw")
#======================

#Light and Dark Mode
def switch_event():
    #customtkinter.set_appearance_mode(color_mode)
    customtkinter.set_appearance_mode(switch.get())
    print("switch toggled, current value:", switch.get())

switch_var = customtkinter.StringVar(value="")
switch = customtkinter.CTkSwitch(app, text="Color Mode", command=switch_event,
variable=switch_var, onvalue="Dark", offvalue="Light")
switch.pack()
#=======================
color_mode = switch.get() 
#=============================


# run loop
app.mainloop()

