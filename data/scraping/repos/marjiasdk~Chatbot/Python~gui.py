from tkinter import Tk, Text, Entry, Button, PhotoImage, END
from openAI_chatbot import Chatbot

def send_message():
    user_message = user_entry.get().strip()

    if user_message:
        chat_log.insert(END, "You: " + user_message + "\n")
        bot_response = Chatbot.get_response(user_message)
        chat_log.insert(END, "Chatbot: " + bot_response + "\n")

    user_entry.delete(0, END)  # Clear the user entry widget

    # Scroll to the bottom of the chat log
    chat_log.see(END)

root = Tk()
root.title("Chatbot powered by OpenAI")
root.geometry("400x500")

# config can have the following parameters: bg, fg, font, padx, pady, etc.
root.config(bg="light blue", padx=7, pady=7)
root.resizable(width=False, height=False)

# Text() is used to display text in multiple lines
chat_log = Text(root, width=50, height=28)
chat_log.pack()

user_entry = Entry(root, width=50, borderwidth=3)
user_entry.pack(side="left", padx=3, pady=(9, 1))

submit_button = Button(root, width=20, text="Send", command=send_message)
submit_button.pack(side="right", padx=3, pady=(9, 1))


root.mainloop()
