import os
import openai
import threading
import tkinter
import tkinter.messagebox
import tkinter.simpledialog
import tkinter.ttk
# Installing the API key and history
openai.api_key = "YOUR API KEY HERE"
history = [
    {"role": "system", "content": "You are ChatGPT by OpenAI. Say hello to the user!"}
]

# Creating a Tkinter Window and adding required elements
root = tkinter.Tk()
root.title("Kiwi for ChatGPT")
root.minsize(300, 300)
root.configure(bg="#a4d0f4")
historylist = tkinter.Listbox(root)
historylist.pack(fill=tkinter.BOTH, expand=True, padx=5, pady=5)
my_scrollbar = tkinter.Scrollbar(
    root, orient=tkinter.HORIZONTAL, command=historylist.xview)
historylist.config(xscrollcommand=my_scrollbar.set)
my_scrollbar.pack(side=tkinter.TOP, fill=tkinter.X)
text = tkinter.Entry(root)
text.pack(side=tkinter.BOTTOM, fill=tkinter.X,
          expand=False, padx=5, pady=5, ipady=5)
text.bind("<Return>", lambda event: submitbtn.invoke())
submitbtn = tkinter.Button(
    root, text="Send", relief="flat", background="#008eff", fg="white")
submitbtn.pack(side=tkinter.BOTTOM, fill=tkinter.X, padx=5, pady=5)
submitbtn.bind("<Return>", lambda event: submitbtn.invoke())
if os.name == "nt":
    # Set icon if it's Windows
    root.iconbitmap(os.getenv("systemdrive") + "\Windows\HelpPane.exe")


# Change API key
def api_key():
    new_api = tkinter.simpledialog.askstring(
        "New API Key", "Change your API key", initialvalue=os.getenv("kiwiapikey"))
    if new_api is not None:
        openai.api_key = new_api
        try:
            result = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Say 'Yes, it works!'"}
                ]
            )
            answer = str(result.choices[0].message["content"])
            tkinter.messagebox.showinfo("New API key", answer)
        except Exception as e:
            tkinter.messagebox.showerror("Error", str(e))


# Clear all chat
def clear_all():
    global history
    getuserreq = tkinter.messagebox.askyesno(
        "New request",
        "Are you sure you want to delete the entire chat?\nAlong with the chat, ChatGPT will erase memory and all permissions.\nAre you sure you want to delete the chat?")
    if getuserreq:
        history = [
            {"role": "system", "content": "You are ChatGPT by OpenAI. Say hello to the user!"}
        ]
        historylist.delete(0, tkinter.END)


def getResponseGPT():

    global history
    while True:
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=history,
                max_tokens=150
            )
            answer = completion.choices[0].message["content"]
            root.after(0,lambda: update_all(answer))
            break
        except openai.error.RateLimitError:
            root.after(0,lambda: update_all(error="The message limit has been reached. Waiting 10 seconds..."))
            import time
            for i in range(11):
                time.sleep(1)
            continue
        except Exception as e:
            root.after(0,lambda: update_all(error=str(e)))
            break

def update_all(last_msg="", error=""):
    global history
    if error.strip() == "":
        history.append({"role": "assistant", "content": last_msg})
        if "\n" in last_msg:
            multiline_array = last_msg.split("\n")
            historylist.insert(
                tkinter.END, "ChatGPT: {}".format(str(multiline_array[0])))
            for element in multiline_array:
                if element != multiline_array[0] and element.strip() != "```":
                    historylist.insert(tkinter.END, "         " + element)
        else:
            historylist.insert(
                tkinter.END, "ChatGPT: {}".format(str(last_msg)))
    else:
        historylist.insert(tkinter.END, "[ERR] " + error.strip())
    submitbtn.config(state="normal")
    text.config(state="normal")

# Send a message
def submitmsg():
    if not text.get().strip() == "":
        global history
        submitbtn.config(state="disabled")
        history.append({"role": "user", "content": text.get().strip()})
        historylist.insert(tkinter.END, "User: {}".format(text.get().strip()))
        text.delete(0, tkinter.END)
        text.config(state="disabled")
        threading.Timer(1.5, getResponseGPT).start()


# Top Menu
menubar = tkinter.Menu(root)
filemenu = tkinter.Menu(menubar, tearoff=0)
filemenu.add_command(label="Start new", command=clear_all)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.quit)
settingsmenu = tkinter.Menu(menubar, tearoff=0)
settingsmenu.add_command(label="Change API key", command=api_key)

menubar.add_cascade(label="Chat", menu=filemenu)
menubar.add_cascade(label="Settings", menu=settingsmenu)
root.config(menu=menubar)

submitbtn.config(command=submitmsg)
root.mainloop()
