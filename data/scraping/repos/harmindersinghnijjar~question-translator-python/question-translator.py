"""1. Write a question into a gui in English.
   2. Send question to gpt3 using openai api.
   3. Recieve answer and translate to Punjabi before outputting the answer to the gui."""

# Required libraries
import tkinter as tk
from tkinter import *
import openai
import googletrans

# OpenAI API Keys
openai.api_key = "xx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Create GUI
root = tk.Tk()
root.title("Question Translator")
root.geometry('400x300')

# Input field
label = Label(root, text="ਅੰਗਰੇਜ਼ੀ ਵਿੱਚ ਆਪਣਾ ਸਵਾਲ ਦਰਜ ਕਰੋ:")
label.pack(pady=10)
question = Entry(root)
question.pack(padx=10, pady=10)

# Submit button
def submit():
    # Get user input question
    q = question.get()
    
    # Send to OpenAI
    response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=q,
    max_tokens=100,
    temperature=0.7,
    frequency_penalty=0.2,
    top_p=0.9
    )
    
    # Get answer
    answer_text = response['choices'][0]['text']
    
    # Translate answer to Punjabi
    translator = googletrans.Translator()
    translated_answer = translator.translate(answer_text, dest='pa')
    
    # Output answer to gui
    answer.insert(0, translated_answer.text)



btn = Button(root, text="ਜਮ੍ਹਾਂ ਕਰੋ", command=submit)
btn.pack(padx=10, pady=10)

# Output field
answer_label = Label(root, text="ਪੰਜਾਬੀ ਵਿੱਚ ਜਵਾਬ:")
answer_label.pack(pady=10)
answer = Entry(root,font=("none", 25), width=40)
answer.place(height=40, width=100)
answer.pack(padx=10, pady=10)

# answer = tk.Text(root, bg="#FFFFFF", height="8", width="50", font="Arial")
# answer.configure(state="disabled")
# answer.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

# # Display the bot's response
# answer.configure(state="normal")
# answer.insert("end", response+"\n\n")
# answer.configure(state="disabled")
# entry.delete(0, tk.END)


root.mainloop()

