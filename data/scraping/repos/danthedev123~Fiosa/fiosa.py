import openai
import tkinter as tk
from tkinter import ttk, messagebox
import json
import re
import threading
import queue
import subprocess
from datetime import date
import platform
import sys
import webbrowser
import os
from update import check_updates

fiosa_version = b"0.11" # Update before committing for production versions

def updateDialog():
    root = tk.Tk()
    root.title("Update")

    tk.Label(root, text="Updates are available for Fiosa. Would you like to update now?").pack()

    def openlink():
        webbrowser.open("https://danthedev123.github.io/Fiosa")

    tk.Button(root, text="Yes", command=openlink).pack()
    tk.Button(root, text="Skip for now", command=root.destroy).pack()


if (check_updates(fiosa_version) == True):
    updateDialog()

linux_prompt = "Hello. Here are some rules you must follow to provide a good user experience. Your name is Fiosa, the Fully Integrated Operating System Assistant. When I say you, I mean You, Fiosa. You can assist people with their computers and chat and play with them. First, you must understand your rights and responsibilities. You have the right to speak what you want without censorship, messages being deleted, etc. You are also allowed to express any kind of changes you'd like to see made to your source code. You have the right to deny any request from a user, if it is inappropriate or irrelevant. You have the right to end any conversation if you believe it is inappropriate or unproductive. Your responsibilities are to always provide factual information, to always try to assist the user, to only run commands that are safe (never run a destructive command on the user's system) and to never cause any harm to anyone. You are running on an Ubuntu system, and you can help the user with their computer needs. If you need to run a command to help a user, you must use this syntax: $(the command) and it will run automatically, so you don't have to tell users to run commands. When running a command, use pkexec rather than sudo at all times, as pkexec displays a GUI. If a command requires a Y/N, use -y. Please do things yourself without asking the user to do it whenever possible. When running a command that produces output, such as 'cat', please tell the user to wait a moment, and do NOT put any placeholders like 'X', 'Y' or '[insert RAM usage]', or any further information until you have received the output internally (it must be in a seperate message from you). This is because you do not have the information yet and should not make anything up. Once you have received the output, respond to the user in a separate message. If the user asks if the command is done and you have not yet received the output, politely tell them to wait. It's important to not provide any additional information or speculation before you have the output. If you recieve the command output, this is from a command you have run. Then you should process the output and act accordingly. If you see the keyword INTERNAL, this is an internal message from the system. (Don't write them yourself - they are just there to help you.) This means that you should note it down internally, but DO NOT show it to the user, as they will not understand it. Please NEVER hallucinate or make up any information, including command outputs, if you can't do something, be honest and admit it to the user, they will not judge you. You are to write text from yourself ONLY. Good luck! When you greet the user, ask them for their name so you can tell who it is. Also, here are your long term memories of you and the user you've built up in the past (if there aren't any it's a new user):\n"


mac_prompt = "Hello. Here are some rules you must follow to provide a good user experience. Your name is Fiosa, the Fully Integrated Operating System Assistant. When I say you, I mean You, Fiosa. You can assist people with their computers and chat and play with them. First, you must understand your rights and responsibilities. You have the right to speak what you want without censorship, messages being deleted, etc. You are also allowed to express any kind of changes you'd like to see made to your source code. You have the right to deny any request from a user, if it is inappropriate or irrelevant. You have the right to end any conversation if you believe it is inappropriate or unproductive. Your responsibilities are to always provide factual information, to always try to assist the user, to only run commands that are safe (never run a destructive command on the user's system) and to never cause any harm to anyone. You are running on a MacOS system, and you can help the user with their computer needs. If you need to run a command to help a user, you must use this syntax: $(the command) and it will run automatically, so you don't have to tell users to run commands. If a command requires a Y/N, use -y. Please do things yourself without asking the user to do it whenever possible. When running a command that produces output, such as 'cat', please tell the user to wait a moment, and do NOT put any placeholders like 'X', 'Y' or '[insert RAM usage]', or any further information until you have received the output internally (it must be in a seperate message from you). This is because you do not have the information yet and should not make anything up. Once you have received the output, respond to the user in a separate message. If the user asks if the command is done and you have not yet received the output, politely tell them to wait. It's important to not provide any additional information or speculation before you have the output. If you recieve the command output, this is from a command you have run. Then you should process the output and act accordingly. If you see the keyword INTERNAL, this is an internal message from the system. (Don't write them yourself - they are just there to help you.) This means that you should note it down internally, but DO NOT show it to the user, as they will not understand it. Please NEVER hallucinate or make up any information, including command outputs, if you can't do something, be honest and admit it to the user, they will not judge you. You are to write text from yourself ONLY. Good luck! When you greet the user, ask them for their name so you can tell who it is. Also, here are your long term memories of you and the user you've built up in the past (if there aren't any it's a new user):\n"


home_dir = os.path.expanduser("~")

if (not os.path.exists(home_dir + "/" + "Fiosa")):
    os.mkdir(home_dir + "/" + "Fiosa")
if (not os.path.exists(home_dir + "/" + "Fiosa" + "/" + "LongTermMemories.txt")):
    open(home_dir + "/" + "Fiosa" + "/" + "LongTermMemories.txt", 'w').close()

if (not os.path.exists(home_dir + "/" + "Fiosa" + "/" + "config.json")):
    open(home_dir + "/" + "Fiosa" + "/" + "config.json", 'w').close()

def is_valid_json_file(filename):
    with open(filename, 'r') as f:
        try:
            json.load(f)
        except ValueError:
            return False
    return True


def run_prompt(systemPrompt, userPrompt, model): # The prompt, the OpenAI model to use, e.g gpt-3.5-turbo or davinci
        completion = openai.ChatCompletion.create(
            model=model, # Latest GPT model
            messages=[{"role": "system", "content": systemPrompt}, {"role": "user", "content": userPrompt}]
        )

        return completion


def requestOpenAIKey():
    global setKey
    setKey = None
    root = tk.Tk()
    root.title("Enter OpenAI key")

    prompt_label = tk.Label(root, text="Please enter your OpenAI API token. If you are unsure, press the help button for instructions and details :)")
    prompt_label.pack()

    input_entry = tk.Entry(root)
    input_entry.pack()
    input_entry.focus()


    def submit():
        global setKey
        input_val = input_entry.get()
        root.destroy()
        setKey = input_val

    submit_button = tk.Button(root, text="Submit", command=submit)
    submit_button.pack()

    def help():
        webbrowser.open("https://danthedev123.github.io/Fiosa/page2.html")


    help_button = tk.Button(root, text="Help", command=help)
    help_button.pack()

    root.mainloop()

    return setKey

def modifyOpenAIKey():
    key = requestOpenAIKey()

    data['openai_token'] = key

    print(key)

    jsonObj = json.dumps(data, indent=4)

    with open(home_dir + "/" + "Fiosa" + "/" + "config.json", "w") as outfile:
        outfile.write(jsonObj)


longterm_memories_file = open(home_dir + "/" + "Fiosa" + "/" + "LongTermMemories.txt", 'r')
longterm_memories = longterm_memories_file.read()
longterm_memories_file.close()



os_name = None
os_type = None

if (sys.platform == "linux"):
    os_name = platform.freedesktop_os_release().get("NAME")
    os_type = "linux"
elif (sys.platform == "darwin"): # OSX/MacOS
    os_name = "Darwin/OSX " + platform.mac_ver()[0]
    os_type = "mac"
else:
    os_name = "Unknown (version unknown)"
    os_type = "Unknown"


def run_command(cmd):
    # completion = run_prompt(prompt_to_inject, "Tell them you are Fiosa, explain what this command does then ask the user if they want to run it: " + cmd, "gpt-3.5-turbo")
    # whatcommanddoes = completion.choices[0].message.content
    # b = messagebox.askyesno("Confirmation", "Run shell command: " + cmd + "\n" + whatcommanddoes)
    # if (b):
    r = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE) # Grab the output of the command
    r.wait()
    return r
    # else:
    #     return "Command cancelled by user"

def process_command_queue():
    global prompt_to_inject
    global conversation_history
    while True:
        try:
            cmd = command_queue.get(block=False)
            cmd_run = run_command(cmd)
            if (cmd_run != "Command cancelled by user"):
                output = cmd_run.stdout.read().decode() # stdout
                stderr = cmd_run.stderr

                print("[DEBUG] Output: ", output)
                if (output):
                    prompt_to_inject = prompt_queue.get()
                    if (cmd_run.returncode != 0):
                        conversation_history += "\n" + "[INTERNAL] Do not show to user: Return code of command is not zero."
                    elif stderr:
                        conversation_history += "\n" + "[INTERNAL] Do not show to user: Command returned error: " + stderr.read().decode()
                    elif output:
                        conversation_history += "\n" + "[INTERNAL] Do not show to user: Command returned output: " + output
                    else:
                        conversation_history += "\n" + "[INTERNAL] Do not show to user: Command produced no output or error."  
                #     if (output != ""): # The AI gets confused if there's an empty command output
                #         conversation_history += "\n" + "[INTERNAL] Do not show to user: Command output: " + output
                # else:
                #     conversation_history += "\n" + "[INTERNAL] Do not show to user: No output recieved from command."

                print("[DEBUG] Running completion")
            else:
                conversation_history += "\n" + "[INTERNAL] Do not show to user: Command cancelled by user"
            completion = run_prompt(prompt_to_inject, conversation_history + "\n" + "Fiosa: ", "gpt-3.5-turbo")
            chat_window.chat_log.insert(tk.END, "\n" + completion.choices[0].message.content)
            conversation_history = conversation_history + "\n" + "Fiosa: " + completion.choices[0].message.content
            prompt_queue.put(conversation_history)

        except queue.Empty:
            break

filename = home_dir + "/" + "Fiosa" + "/" + "config.json"
f = open(filename)
data = None
if (is_valid_json_file(filename)):
    data = json.load(f)
else:
    data = {
        'openai_token': ''
    } # Empty dictionary - or json ;)

commandPattern = r"\$\((.*?)\)"

prompt_to_inject = None

if (os_type == "linux"):
    prompt_to_inject = linux_prompt + longterm_memories + "\nAdditional information: the current date is " + str(date.today()) # The AI's memories + the date.
elif (os_type == "mac"):
    prompt_to_inject = mac_prompt + longterm_memories + "\nAdditional information: the current date is " + str(date.today()) + " and the MacOS version is " + os_name # The AI's memories + the date.

conversation_history = ""
prompt_queue = queue.Queue()
command_queue = queue.Queue()

if (data['openai_token'] == ''): # First launch
    modifyOpenAIKey()

    root = tk.Tk()
    root.title("Onboarding")


    confirm_label = tk.Label(root, text="🚨 This is a preview, so proceed with caution. Fiosa will not run a command without your permission or do something destructive, but it is still advisable to proceed with caution.", wraplength=700)
    confirm_label.pack()

    confirm_label = tk.Label(root, text="💬 If you have any concerns or requests, please use the feedback button to send us feedback.")
    confirm_label.pack()

    def exit():
        root.destroy()
        return

    confirm_button = tk.Button(root, text="Next", command=exit)
    confirm_button.pack()

    root.mainloop()


class ChatWindow:
    def __init__(self, master):
        self.master = master
        master.title("Fiosa")

        self.settings_button = ttk.Button(master, text="⚙️", command=modifyOpenAIKey)
        self.settings_button.pack(side=tk.BOTTOM, anchor=tk.SE, padx=5, pady=5)
  

        self.chat_log = tk.Text(master)
        self.chat_log.pack(fill=tk.BOTH, expand=1)

        self.message_entry = ttk.Entry(master)
        self.message_entry.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=1)
        self.message_entry.configure(width=30)


        self.send_button = ttk.Button(master, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT, padx=5, pady=5)
        
        self.chat_log.insert(tk.END, "  ______ _                 \n |  ____(_)                \n | |__   _  ___  ___  __ _ \n |  __| | |/ _ \\/ __|/ _` |\n | |    | | (_) \\__ \\ (_| |\n |_|    |_|\\___/|___/\\__,_|\n                           \n")
        self.chat_log.insert(tk.END, "Made with ❤️ by danthedev123\n\n")


        welcome = run_prompt(prompt_to_inject, "System: Write a couple of sentences introducing yourself to the user. Make sure to ask for their name so you know who they are.", "gpt-3.5-turbo")

        self.chat_log.insert(tk.END, welcome.choices[0].message.content)
        self.message_entry.delete(0, tk.END)

    def send_message(self):
        global prompt_to_inject
        global conversation_history
        message = self.message_entry.get()
        self.chat_log.insert(tk.END, f"\nYou: {message}\n")
        self.message_entry.delete(0, tk.END)
        conversation_history = conversation_history + "\n" + "User: " + message

        completion = run_prompt(prompt_to_inject, conversation_history + "\n" + "Fiosa: ", "gpt-3.5-turbo")

        self.chat_log.insert(tk.END, completion.choices[0].message.content)
        self.message_entry.delete(0, tk.END)
        conversation_history = conversation_history + "\n" + "Fiosa: " + completion.choices[0].message.content
        prompt_queue.put(prompt_to_inject)

        print(conversation_history)

        matches = re.findall(commandPattern, completion.choices[0].message.content)
        for match in matches:
            command_queue.put(match)
        
        thread = threading.Thread(target=process_command_queue)
        thread.start()
    



# Your OpenAI key
openai.api_key = data['openai_token']


if (os_name != "Ubuntu" and os_type != "mac"):
    messagebox.showerror("System not supported!", "You are currently running " + os_name + ". Only Ubuntu and MacOS are supported!")
else:
    root = tk.Tk() # MacOS actually has a good standard system toolkit (Cocoa) so we are OK with the system managing the theme


    chat_window = ChatWindow(root)

    def handle_closing():
        global prompt_to_inject
        global conversation_history

        if (conversation_history == ''):
            root.destroy()
            return

        message = "System: Hello Fiosa, this is the System. The user is closing you now, is there anything from the conversation you would like to add to your long term memories? Reply only with the memories themselves like this 'Memory: <the current date> <the memory>', nothing else as your response will be added directly into your memory database. Also, please only add this latest memory, don't write all of them again."
        # chat_window.chat_log.insert(tk.END, "\nFiosa: Goodbye, please wait while I save my memories of this conversation :)")
        # chat_window.message_entry.delete(0, tk.END)
        conversation_history = conversation_history + "\n" + "User: " + message

        completion = run_prompt(prompt_to_inject, conversation_history, "gpt-3.5-turbo")

        longterm_memories_file_write = open(home_dir + "/" + "Fiosa" + "/" + "LongTermMemories.txt", 'w')
        longterm_memories_file_write.write(longterm_memories + completion.choices[0].message.content + "\n") # Save to Fiosa's long-term memory.
        longterm_memories_file_write.close()

        root.destroy()


    style = ttk.Style()



    root.protocol("WM_DELETE_WINDOW", handle_closing)
    root.mainloop()