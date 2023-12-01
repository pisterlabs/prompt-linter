from tkinter import *
import openai
import re
import requests.exceptions
from openai.error import AuthenticationError, InvalidRequestError, APIConnectionError, APIError, RateLimitError

openai.api_key = "YOUR_API_KEY"

chatbot_engine = "text-davinci-002"

root = Tk()
root.title("Svxy AI")
root.iconbitmap("./assets/icon.ico")
root.geometry("600x580")
root.configure(bg="black")
root.resizable(False, False)

bg_color = "#210000"
text_color = "#FF0000"
btn_color = "#302a2a"

response_box = Text(root, width=75, height=30, bg=bg_color, fg=text_color)
response_box.insert(END, "\nSvxy:\nWelcome to the chat! What can I help you with?\n")
response_box.config(state=DISABLED)
response_box.pack(padx=10, pady=10)

input_box = Entry(root, width=60, bg=bg_color, fg=text_color)
input_box.pack(side=BOTTOM, padx=10, pady=10)

def generate_response(prompt):
    try:
        response = openai.Completion.create(
            engine=chatbot_engine,
            prompt=prompt,
            max_tokens=4000,
            n=1,
            stop=None,
            temperature=0.7,
        )
        response_text = response.choices[0].text.strip()
        response_text = re.sub('[^0-9a-zA-Z\n\.\?!@#\$%&\*\(\)\-_\+=<>\[\]\{\}\|\^~`"":;,\/\\\'\\\]+', ' ', response_text)
        return response_text
    except (ValueError, TypeError, AttributeError, ZeroDivisionError, FileNotFoundError, IOError, KeyboardInterrupt, MemoryError, requests.exceptions.RequestException, AuthenticationError, InvalidRequestError,APIConnectionError, APIError, RateLimitError) as e:
        response_box.config(state=NORMAL)
        response_box.insert(END, "\n\nSvxy: Sorry, there was an error processing your request. Please try a different request...\n\n")
        response_box.config(state=DISABLED)

def send_message():
    user_input = input_box.get()
    input_box.delete(0, END)

    prompt = "User: {}\n\nSvxy:\n\n".format(user_input)
    response_text = generate_response(prompt)

    if response_text:
        response_box.config(state=NORMAL)
        response_box.insert(END, "\nYou:\n" + user_input + "\n")
        response_box.insert(END, "\nSvxy:\n" + response_text + "\n")
        response_box.config(state=DISABLED)
        response_box.see(END)

send_button = Button(root, text="Send", command=send_message, bg=btn_color, fg=text_color)
send_button.pack(pady=(4.5, 0))

input_box.focus()

root.mainloop()