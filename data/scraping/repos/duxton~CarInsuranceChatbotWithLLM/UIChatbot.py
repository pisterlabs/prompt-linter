from tkinter import *
import json
import openai

# GUI
root = Tk()
root.title("Chatbot")

BG_GRAY = "#FFFFFF"
BG_COLOR = "#FFFFFF"
TEXT_COLOR = "#000000"

FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

api_key = "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"
openai.api_key = api_key
# Send function
def send():
    # send is user input
    send = "Me: " + e.get()
    # txt is respond from gpt 
    txt.insert(END, "\n" + send)
    # Call GPT3 model and store the response to a variable and return 
    # Replace with GPT response 
    #new_prompt = "My Car proton  model x70 year 2021 ncd 25%, how much would my car insurance be?"
    new_prompt = e.get() + " ->"
    if "FAQ-> " in e.get(): 
        # for FAQ questions 
        replaced_prompt = new_prompt.replace("FAQ-> ", '')
        answer = openai.Completion.create(
            model="davinci:ft-pogtdev-2023-06-11-07-19-08",
            prompt=replaced_prompt,
            temperature=1,
            max_tokens=100,
            top_p=1,
            best_of=5,
            frequency_penalty=1,
            presence_penalty=1
        )
    else:
        # For car price 
        replaced_prompt = new_prompt.replace("Car-> ", '')
        answer = openai.Completion.create(
            model="davinci:ft-pogtdev-2023-06-11-07-19-08",
            prompt=replaced_prompt,
            temperature=1,
            max_tokens=20,
            top_p=1,
            best_of=5,
            frequency_penalty=1,
            presence_penalty=1
        )

    txt.insert(END, "\n" + "POIBOT: " + answer['choices'][0]['text'])
    # https://www.geeksforgeeks.org/gui-chat-application-using-tkinter-in-python/
    # get user input
    user = e.get().lower()
 
    e.delete(0, END)
 

# Welcome on the top
lable1 = Label(root, bg=BG_COLOR, fg=TEXT_COLOR, text="Welcome", font=FONT_BOLD, pady=10, width=55, height=1).grid(
	row=0)

# Display text
txt = Text(root, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT, width=60)
txt.grid(row=1, column=0, columnspan=2)
txt.insert(END, "\n" + "POIBOT: Hi Welcome, how may I help you? Please include FAQ or Car to get a better result")

# Scroll bar
scrollbar = Scrollbar(txt)
scrollbar.place(relheight=1, relx=0.974)

# Input from user
e = Entry(root, bg="#FFFFFF", fg=TEXT_COLOR, font=FONT, width=55)
e.grid(row=2, column=0)

# Sent Btn
send = Button(root, text="Send", font=FONT_BOLD, bg=BG_GRAY,
			command=send).grid(row=2, column=1)

root.mainloop()
