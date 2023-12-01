#JMR OpenAI GUI Select & Temp SimpleChat with Streaming July 5 2023  # cut and paste to the menu!
import os
import tkinter as tk
from tkinter import scrolledtext
import time
import platform
import openai
from tkinter import Menu, END
print(platform.machine())


API_KEY = input ("enter your API AI key: ")

openai.api_key = API_KEY
model_name  = "gpt-3.5-turbo"
#model_name = "gpt-4"
temp = 0

def clear_text(widget):
    if widget.winfo_class() == 'Text':
        widget.delete('1.0', END)
    elif widget.winfo_class() == 'Entry':
        widget.delete(0, END)

def show_context_menu(event):
    context_menu = Menu(root, tearoff=0)
    context_menu.add_command(label="Cut", command=lambda: root.focus_get().event_generate("<<Cut>>"))
    context_menu.add_command(label="Copy", command=lambda: root.focus_get().event_generate("<<Copy>>"))
    context_menu.add_command(label="Paste", command=lambda: root.focus_get().event_generate("<<Paste>>"))
    context_menu.add_command(label="Clear", command=lambda: clear_text(root.focus_get()))
    context_menu.tk_popup(event.x_root, event.y_root)  

def set_model_name(name):
    global model_name
    model_name = name
    temp = tempLLM.get()
    root.title(("JMR's Little " + model_name + " Chat. Temp: " + str(temp))) # Set the title for the window


def save_text():
    prompt = entry.get("1.0", "end-1c")  # Get text from Text widget
    generated_text = text_area.get("1.0", "end-1c")  # Get text from Text widget

    # Create the filename using the first 10 characters of the prompt and a 4-digit timestamp
    filename = prompt[:10] + "_" + time.strftime("%m%d-%H%M") + ".txt"

    # Create a directory to save the files if it doesn't exist
    directory = "saved_texts"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the prompt and generated text to the file
    with open(os.path.join(directory, filename), "w") as file:
        file.write("Prompt:\n" + prompt + "\n\nGenerated Text:\n" + generated_text)
    print("Text saved successfully in: ",filename)


def talk_to_LLM():
    prompt = entry.get("1.0", "end-1c") # get text from Text widget
    response =  (openai.ChatCompletion.create(
      model= model_name,
      messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ], max_tokens = int(max_new_tokens.get()), temperature = temp, stream = True
    ))

    final = ""
    print (response) # a function when streaming.
    #content = response["choices"][0]["message"]["content"] # when not streaming
    text_area.insert(tk.END, f"\nResult: ")
    for stream in response:
        text = stream["choices"][0]["delta"].get("content", "") # Format for openAI
        #text = stream['choices'][0]['text']
        text_area.insert(tk.END, text)
        text_area.see(tk.END)  # Make the last line visible
        text_area.update()
        final = final + text 
    text_area.insert(tk.END, "\n")
    text_area.update() 
    print (final)

root = tk.Tk()
root.title(("JMR's Little " + model_name + " Chat. Temp: " + str(temp))) # Set the title for the window
root.geometry("800x600")

root.bind("<Button-2>", show_context_menu)

root.grid_rowconfigure(0, weight=1) # Entry field takes 1 part
root.grid_rowconfigure(1, weight=0) # "Send" button takes no extra space
root.grid_rowconfigure(2, weight=3) # Output field takes 3 parts
root.grid_columnconfigure(0, weight=1) # Column 0 takes all available horizontal space

entry = scrolledtext.ScrolledText(root, height=5, wrap="word")
entry.grid(row=0, column=0, columnspan=8, sticky='nsew')

button = tk.Button(root, text="Send", command=talk_to_LLM)
button.grid(row=1, column=7, sticky='e')

button_3_5 = tk.Button(root, text="GPT3.5", command=lambda: set_model_name("gpt-3.5-turbo"))
button_3_5.grid(row=1, column=5, columnspan=1)

button_4 = tk.Button(root, text="GPT4", command=lambda: set_model_name("gpt-4"))
button_4.grid(row=1, column=6, columnspan=1)


max_label = tk.Label(root, text="Max New Tokens:")
max_label.grid(row=1, column=3, sticky='w')

max_new_tokens = tk.DoubleVar(value = 256)
slider_token = tk.Scale(root, from_=0, to=8000, resolution=10, orient=tk.HORIZONTAL, variable=max_new_tokens)
slider_token.grid(row=1, column=4, sticky='w')


temp_label = tk.Label(root, text="Temperature:")
temp_label.grid(row=1, column=1, sticky='w')

tempLLM = tk.DoubleVar()
slider = tk.Scale(root, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, variable=tempLLM)
slider.grid(row=1, column=2, sticky='w')
temp = tempLLM.get()

save_button = tk.Button(root, text="Save", command=save_text)
save_button.grid(row=1, column=0, sticky='w')

text_area = tk.Text(root, wrap="word")
text_area.grid(row=2, column=0, columnspan=8, sticky='nsew')

# Adding a scrollbar to the text area
scrollbar = tk.Scrollbar(text_area)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
text_area.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=text_area.yview)
root.mainloop()

