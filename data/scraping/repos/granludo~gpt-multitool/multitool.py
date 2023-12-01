import openai, json
import tkinter as tk
from tkinter import ttk  # ttk is used for styling
from tkinter import filedialog, scrolledtext
import threading


# Multitool.py 
# by Marc Alier 2023
# This script uses the OpenAI API to process file with GPT models breaking it down in chunks
# licensed under the GNU General Public License v3.0
# you need to have a key to use the OpenAI API
# you can get one here: https://beta.openai.com/docs/developer-quickstart/api-key
# this program looks for the key in a file stated in the mykeypath variable 

mykeypath='..//..//mykey.json'
## sample content for the file mykey.json
# {
#  "key": "copy_your_key_here"
# }


MODEL = "gpt-4" # this model is way more expensive

MODEL_OPTIONS = ["gpt-3.5-turbo", "gpt-4"]  # options for OpenAI models

# Load prompts
with open("prompts.json") as fp:
    data = json.load(fp)
    prompts = {prompt['name']: {'system': prompt['system'], 'user': prompt['user']} for prompt in data['prompts']}

# Function to load a prompt
def load_prompt():
    selected_prompt = prompt_var.get()
    system_prompt_text.delete('1.0', tk.END)
    system_prompt_text.insert('1.0', prompts[selected_prompt]['system'])
    user_prompt_text.delete('1.0', tk.END)
    user_prompt_text.insert('1.0', prompts[selected_prompt]['user'])

# Function to save a modified prompt
def save_prompt():
    selected_prompt = prompt_var.get()
    prompts[selected_prompt] = {'system': system_prompt_text.get('1.0', tk.END), 'user': user_prompt_text.get('1.0', tk.END)}
    with open("prompts.json", "w") as fp:
        json.dump({'prompts': [{'name': k, 'system': v['system'], 'user': v['user']} for k, v in prompts.items()]}, fp)

def copy_edit(text):
    query=f"{text}"
    print_status("\n\n\n enviant a openai:-----------------------------------------------------------------------------------------------")
    print_status("Query: "+query)

    try:
        response = openai.ChatCompletion.create(
            model=model_var.get(),  # use the selected model
            messages=[
                {
                    "role": "system",
                    "content": prompt_text.get("1.0", tk.END)
                },
                {"role": "user", "content": query},
            ],
            max_tokens=max_tokens_var.get(),
            n=1,
            stop=None,
            temperature=0,
        )
    except Exception as e:
        print_status(f"Error: {e}")
        return ""  # return an empty string if an error occurs

    print_status("\n\n\nOpenai a respon:-----------------------------------------------------------------------------------------------")
    print_status(response['choices'][0]['message']['content'])

    return response['choices'][0]['message']['content']


def grammar_rev(chunk_size=80):
    i=0
    text=text_area1.get('1.0', tk.END).split("\n")
    outuput_text=""
    text_area2.delete('1.0', tk.END)
    while i < len(text):
        update_gui("linia:"+str(i))
        edited_text = copy_edit("\n".join(text[i:i+chunk_size]))
        outuput_text += edited_text  # append the edited text
        i += chunk_size
    text_area2.insert(tk.END, outuput_text)  # update the GUI after the loop
    update_gui("Procés acabat")
    
def print_status(message):
    root.after(0, update_gui, message)
    

def load_file():
    file_path = filedialog.askopenfilename(filetypes=[('Text Files', '*.txt'), ('Markdown Files','*.md') ])    
    if not file_path:
        return
    text_area1.delete('1.0', tk.END)
    try:
        with open(file_path, 'r') as file:
            text_area1.insert(tk.END, file.read())
    except Exception as e:
        print_status(f"Error: {e}")

def get_text():
    return text_area.get('1.0', tk.END)

def save_file():
    file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[('Text Files', '*.txt')])
    if not file_path:
        return
    try:
        with open(file_path, 'w') as file:
            file.write(text_area2.get('1.0', tk.END))
    except Exception as e:
        print_status(f"Error: {e}")


def update_gui(message):
    status_text.insert(tk.END, message + '\n')
    status_text.see(tk.END)


def start_process():
    process_thread = threading.Thread(target=grammar_rev, args=(chunk_size_var.get(),))
    process_thread.start()

root = tk.Tk()
# Create a top frame for the buttons
button_frame = tk.Frame(root)
button_frame.pack(fill=tk.X, side=tk.TOP)

# Create buttons
some_button = tk.Button(button_frame, text='Run process', command=lambda: start_process())
some_button.pack(side=tk.LEFT)
load_button = tk.Button(button_frame, text='Load file', command=load_file)
load_button.pack(side=tk.LEFT)
save_button = tk.Button(button_frame, text='Save file', command=save_file)
save_button.pack(side=tk.LEFT)

# Create a drop-down menu to select the model
model_var = tk.StringVar(root)
model_var.set(MODEL_OPTIONS[0])  # set the default option
model_label = tk.Label(button_frame, text='Select Model:')
model_label.pack(side=tk.LEFT)
model_menu = ttk.Combobox(button_frame, textvariable=model_var, values=MODEL_OPTIONS)
model_menu.pack(side=tk.LEFT)

# Create a Spinbox to select the chunk size
chunk_size_var = tk.IntVar(root)
chunk_size_var.set(80)  # set the default chunk size
chunk_size_label = tk.Label(button_frame, text='Chunk Size:')
chunk_size_label.pack(side=tk.LEFT)
chunk_size_spinbox = tk.Spinbox(button_frame, from_=1, to=500, textvariable=chunk_size_var)
chunk_size_spinbox.pack(side=tk.LEFT)

# Create a Spinbox to select the maximum number of tokens
max_tokens_var = tk.IntVar(root)
max_tokens_var.set(4000)  # set the default max tokens
max_tokens_label = tk.Label(button_frame, text='Max Tokens:')
max_tokens_label.pack(side=tk.LEFT)
max_tokens_spinbox = tk.Spinbox(button_frame, from_=1, to=10000, textvariable=max_tokens_var)
max_tokens_spinbox.pack(side=tk.LEFT)


# Create a frame for text areas
text_area_frame = tk.Frame(root)
text_area_frame.pack(fill=tk.BOTH, expand=True, side=tk.TOP)

# Create first text area with a title and scrollbar
frame1 = tk.Frame(text_area_frame)
frame1.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
text_area1_title = tk.Label(frame1, text="Fitxer Origen")
text_area1_title.pack()
scrollbar1 = tk.Scrollbar(frame1)
text_area1 = tk.Text(frame1, wrap='word', width=80, yscrollcommand=scrollbar1.set, bg='white', fg='black')
scrollbar1.pack(side=tk.RIGHT, fill=tk.Y)
text_area1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar1.config(command=text_area1.yview)

# Create second text area with a title and scrollbar
frame2 = tk.Frame(text_area_frame)
frame2.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT)
text_area2_title = tk.Label(frame2, text="Sortida generada")
text_area2_title.pack()
scrollbar2 = tk.Scrollbar(frame2)
text_area2 = tk.Text(frame2, wrap='word', width=80, yscrollcommand=scrollbar2.set, bg='black', fg='white')
scrollbar2.pack(side=tk.RIGHT, fill=tk.Y)
text_area2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar2.config(command=text_area2.yview)

# Create a frame for status messages
status_frame = tk.Frame(root)
status_frame.pack(fill=tk.BOTH, side=tk.BOTTOM)
status_label = tk.Label(status_frame, text='Status Messages:')
status_label.pack()
status_text = scrolledtext.ScrolledText(status_frame, wrap='word', bg='lightgray',fg='black')
status_text.pack(expand=True, fill='both')

# Create a frame for prompt
# Create two frames for system and user prompts
system_prompt_frame = tk.Frame(root)
system_prompt_frame.pack(fill=tk.BOTH, side=tk.BOTTOM)
system_prompt_label = tk.Label(system_prompt_frame, text='System Prompt:')
system_prompt_label.pack()
system_prompt_text = tk.Text(system_prompt_frame, height=3, bg='white', fg='black')
system_prompt_text.pack(fill=tk.X)

user_prompt_frame = tk.Frame(root)
user_prompt_frame.pack(fill=tk.BOTH, side=tk.BOTTOM)
user_prompt_label = tk.Label(user_prompt_frame, text='User Prompt:')
user_prompt_label.pack()
user_prompt_text = tk.Text(user_prompt_frame, height=3, bg='white', fg='black')
user_prompt_text.pack(fill=tk.X)



with open("defaultprompt.json") as fp:
    data = json.load(fp)
    defaultprompt=data['defaultprompt']
    prompt_text.insert("1.0", defaultprompt)

with open(mykeypath, 'r') as f:
    try:
        data = json.load(f)
        print("OK")
    except Exception as e:
        print(f"Error: {e}")

# Initialize the OpenAI API
openai.api_key = data['key']


root.mainloop()

# 
