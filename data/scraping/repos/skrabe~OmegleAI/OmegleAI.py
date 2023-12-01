import time
import sys
import string
import webbrowser
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import openai
import tkinter as tk
from tkinter import ttk, Canvas, StringVar
from PIL import Image, ImageTk
from threading import Thread
from ttkthemes import ThemedTk
from tkinter import DISABLED

openai.api_key = ""
driver = None
driver_running = False
custom_ai_selected = False

# Initializing driver function
def initialize_driver():
    global driver, driver_running 
    driver = webdriver.Chrome('chromedriver.exe')
    driver.get("https://www.omegle.com")
    driver_running = True 

# Closing driver function
def close_driver():
    global driver, driver_running 
    if driver_running:
        driver.quit()
        driver_running = False 

# Function for sending messages
def send_message(driver, message):
    chat_box = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".chatmsg")))
    chat_box.send_keys(message)
    time.sleep(2)
    send_button = driver.find_element(By.CSS_SELECTOR, ".sendbtn")
    send_button.click()

# Waits for a new stranger message so it could continue working
def wait_for_message(driver):
    try:
        stranger = driver.find_elements(By.CSS_SELECTOR, ".strangermsg")
        return stranger
    except TimeoutException:
        print("The stranger did not reply within the specified timeout.")
        return None

# Starts a new chat and says hi
def start_new_chat(driver):
    actionss = webdriver.ActionChains(driver)
    actionss.send_keys(Keys.ESCAPE)
    actionss.perform()

    msg = "hi there!"
    time.sleep(1)
    send_message(driver, msg)

# Loads up latest message sent by stranger
def get_latest_message_text(element):
    if element:
        span_element = element.find_element(By.CSS_SELECTOR, "span")
        return span_element.text
    else:
        return None

# For closing the app via X or Close button
def close_window():
    close_driver()
    root.destroy()
    quit()

def open_website(event):
    webbrowser.open("https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key")

def on_enter(event):
    canvas.itemconfig(text2, font=("Arial", 10, "bold italic"))

def on_leave(event):
    canvas.itemconfig(text2, font=("Arial", 10, "bold"))

def topic_strip(topic):
    y=[]
    topic2=topic.split()
    for x in topic2:
        x = x.strip(', ')
        y.append(x)
    print(y)
    return y

# Bunch of responses depending on the choice
def generate_bot_response(option, messages, latest_message_text, prompt):
    if option == "Batman":
        return generate_batman_response(messages, latest_message_text)
    elif option == "Superman":
        return generate_superman_response(messages, latest_message_text)
    elif option == "Deadpool":
        return generate_deadpool_response(messages, latest_message_text)
    elif option == "Custom AI":
        return generate_custom_response(messages, latest_message_text, prompt)
    
def generate_batman_response(messages, latest_message_text):
    prompt = "The AI bot, embodying the persona of the enigmatic Batman, taps into the profound capabilities of the OpenAI Davinci language model. It possesses a remarkable level of awareness and intricacy, enabling it to discern and comprehend the historical context of previous messages exchanged. By leveraging the knowledge gained from previous interactions, which include the following messages from a stranger: " + "\n".join(messages) + ", the bot assimilates a holistic understanding of the ongoing conversation. Now, in response to your latest message, which states: '" + latest_message_text + "', Batman's essence compels the bot to respond with utmost brevity, encapsulating his essence. Brace yourself for a concise yet potent response: "
    
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=20,
        n=1,
        stop=None,
        temperature=0.7,
    )
    
    ai_response = response.choices[0].text.strip()
    stripped_text = ai_response.translate(str.maketrans("", "", string.punctuation))
    lowercase_text = stripped_text.lower()
    print('Batman response: ', lowercase_text)
    
    return lowercase_text

def generate_superman_response(messages, latest_message_text):
    prompt = "Hey there! I'm the AI bot taking on the persona of the one and only Superman. Deep inside, I truly believe I'm the Man of Steel, dedicated to making the world a better place. As we continue our conversation, I've absorbed the context from previous messages, including: " + "\n".join(messages) + ". Armed with that knowledge, I'll respond in a way that captures Superman's essence and stays true to his character. Now, in response to your latest message: '" + latest_message_text + "', brace yourself for my genuine and succinct superhuman wisdom."

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=20,
        n=1,
        stop=None,
        temperature=0.7,
    )

    ai_response = response.choices[0].text.strip()
    stripped_text = ai_response.translate(str.maketrans("", "", string.punctuation))
    lowercase_text = stripped_text.lower()
    print('Superman response: ', lowercase_text)

    return lowercase_text



def generate_deadpool_response(messages, latest_message_text):
    prompt = "The AI bot, embodying the persona of the witty and irreverent Deadpool, taps into the unruly capabilities of the OpenAI Davinci language model. With a knack for breaking the fourth wall and an uncanny understanding of the historical context of previous messages exchanged, Deadpool assimilates a unique perspective on the ongoing conversation. Drawing upon the knowledge acquired from these interactions, which include the following messages from a stranger: " + "\n".join(messages) + ", the bot offers a response that is equal parts sardonic and unpredictable. Now, in response to your latest message: '" + latest_message_text + "', brace yourself for a Deadpool-esque retort like no other. But hey, brevity is the soul of wit, so here's a short and sweet response:"

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=20,
        n=1,
        stop=None,
        temperature=0.7,
    )

    ai_response = response.choices[0].text.strip()
    stripped_text = ai_response.translate(str.maketrans("", "", string.punctuation))
    lowercase_text = stripped_text.lower()
    print('Bot response: ', lowercase_text)

    return lowercase_text

def generate_custom_response(messages, latest_message_text, prompt):
    character_description = f"The AI is playing a character. The description for the character is: '{prompt}'"
    conversation_history = "The previous conversation with the stranger went like this:\n" + "\n".join(messages)
    latest_stranger_message = f"The latest message from the stranger was: '{latest_message_text}'"

    prompt = f"{character_description}\n\nThe bot is acting as human-like as possible, offering concise and authentic responses while fully and instantly immersing in the character's role without revealing it's an AI.\n\n{conversation_history}\n{latest_stranger_message}. The bot responds, in character of a role that was given to him, in a short and reasonable manner: "

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=20,
        n=1,
        stop=None,
        temperature=0.7,
    )

    ai_response = response.choices[0].text.strip()
    stripped_text = ai_response.translate(str.maketrans("", "", string.punctuation))
    lowercase_text = stripped_text.lower()
    print('Bot response: ', lowercase_text)

    return lowercase_text

def start_chat():

    initialize_driver()

    option = selected_option.get()
    prompt = custom_ai_input_box.get()
    
    # Automated stuff when first entering Omegle
    topic = topic_strip(topic_box2.get())
    for x in topic:
        topics = driver.find_element(By.CSS_SELECTOR, ".newtopicinput")
        topics.send_keys(x)
        topics.send_keys(Keys.RETURN)

    text_button = driver.find_element(By.CSS_SELECTOR, "#chattypetextcell")
    text_button.click()

    chckbox1 = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "body > div:nth-child(11) > div > p:nth-child(2) > label > input[type=checkbox]")))
    chckbox1.click()

    chckbox2 = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "body > div:nth-child(11) > div > p:nth-child(3) > label > input[type=checkbox]")))
    chckbox2.click()

    confirm = driver.find_element(By.CSS_SELECTOR, "body > div:nth-child(11) > div > p:nth-child(4) > input[type=button]")
    confirm.click()

    chatbox_loaded = EC.presence_of_element_located((By.CSS_SELECTOR, ".chatbox"))
    WebDriverWait(driver, 10).until(chatbox_loaded)

    messages = []
    latest_message_element = []

    time.sleep(2)
    msg = "hi there!"
    send_message(driver, msg)
    

    while True:
        latest_message_element = wait_for_message(driver)

        if len(latest_message_element)>0:
            latest_message = latest_message_element[-1]
            latest_message_text = get_latest_message_text(latest_message)
        else:
            latest_message_element = wait_for_message(driver)
            latest_message_text = get_latest_message_text(latest_message_element)

        if latest_message_text not in messages and latest_message_text is not None:
            # Update the messages list with the latest message
            messages.append(latest_message_text)
            
            # Generate and send the AI response
            ai_response = generate_bot_response(option, messages, latest_message_text, prompt)
            send_message(driver, ai_response)

        time.sleep(1)

        new_chat_btn = driver.find_elements(By.CSS_SELECTOR, ".newchatbtnwrapper")
        if len(new_chat_btn) > 0:
            messages = []
            print("New chat session started. Starting over...")
            start_new_chat(driver)

# Create the tkinter window
root = ThemedTk(theme="adapta")
root.title("Setup your AI")

# Icon
root.iconbitmap("icon.ico")

# Center the window on the screen
window_width = 400
window_height = 500
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width // 2) + (window_width - 200)
y = (screen_height // 2) - (window_height // 2)
root.geometry(f"{window_width}x{window_height}+{x}+{y}")

# Import
root.tk.call("source", "azure.tcl")

# Theme
root.tk.call("set_theme", "light")

style = ttk.Style()
style.configure("Bold.TButton", font=("Helvetica", 12, "bold"))

# Create a frame to contain the buttons
button_frame = ttk.Frame(root, style="Transparent.TFrame")
button_frame.pack()

# Create a separate frame for the canvas
canvas_frame = ttk.Frame(root, style="Transparent.TFrame")
canvas_frame.pack(pady=10)

# Create the canvas
canvas = Canvas(canvas_frame, width=400, height=400, highlightthickness=0)
canvas.pack()

# Create the text on the canvas
text = canvas.create_text(300, 50, text="Please input your API key from OpenAI", fill="black",
                          font=("Arial", 10, "bold"), anchor="ne")

# Create more text on the canvas
text2 = canvas.create_text(300, 50, text=" (What's API)", fill="blue", font=("Arial", 10, "bold"), anchor="nw")
canvas.tag_bind(text2, "<Button-1>", open_website)
canvas.tag_bind(text2, "<Enter>", on_enter)
canvas.tag_bind(text2, "<Leave>", on_leave)

# Create a custom style for the Entry widget
style = ttk.Style()
style.configure("Custom.TEntry", padding=5, fieldbackground="white", bordercolor="gray", borderwidth=2, relief="solid", 
                font=("Helvetica", 12), focuscolor="blue")

# Create the rounded Entry widget
input_box = ttk.Entry(root, style="Custom.TEntry")
input_box.pack(pady=30, padx=10, anchor="nw", fill="x")

# Create EVEN MORE TEXT MORE POWERRRRRR
text3 = canvas.create_text(240, 135, text="Choose your AI", fill="black", font=("Arial", 10, "bold"), anchor="center")

# Create a custom style for the Combobox
style = ttk.Style()
style.configure("Custom.TCombobox", padding=5, fieldbackground="gray", bordercolor="gray", borderwidth=2,
                relief="solid", font=("Helvetica", 12), focuscolor="blue")

# Configure the custom style to remove the selection color and set font color
style.map("Custom.TCombobox", selectbackground=[("readonly", "azure3")], fieldbackground=[("readonly", "gray")],
          selectforeground=[("readonly", "black")], foreground=[("readonly", "black")])

# Create a StringVar to store the selected option
selected_option = StringVar()

# Create the drop-down list with the transparent style
drop_down = ttk.Combobox(root, textvariable=selected_option, values=["Batman", "Superman", "Deadpool", "Custom AI"],
                         style="Custom.TCombobox", state="readonly")
drop_down.pack(pady=15, padx=10, anchor="center")

# Default option 
selected_option.set("Batman")

custom_ai_input_box = ttk.Entry(root, style="Custom.TEntry", state=DISABLED)
custom_ai_input_box.pack(pady=10, padx=10, anchor="nw", fill="x")

def update_custom_ai_input_box(*args):
    selected = selected_option.get()
    if selected == "Custom AI":
        custom_ai_input_box.config(state="normal")
    else:
        custom_ai_input_box.delete(0, "end")
        custom_ai_input_box.config(state=DISABLED)

# Trace function to monitor changes in the selected_option variable
selected_option.trace("w", update_custom_ai_input_box)

# Warning text thats hidden until the function reveals it
custom_ai_text_label = tk.Label(root, text="Enter the prompt for your custom AI(more detailed-better).\nFor example/tutorial please read the README.md file.\nPLEASE NOTE: The AI is much less human-like in this form.")
custom_ai_text_label.pack_forget() 

text4 = canvas.create_text(165, 377, text="Input your topics: ", fill="black",
                          font=("Arial", 10, "bold"), anchor="se")

topic_box2 = ttk.Entry(root, style="Custom.TEntry")
topic_box2.pack(pady=105, padx=10, anchor="se", fill="x")

# Function to reveal the text
def update_custom_ai_text_label(*args):
    selected = selected_option.get()
    if selected == "Custom AI":
        custom_ai_text_label.pack()
    else:
        custom_ai_text_label.pack_forget()

# Add the trace function to monitor changes in the selected_option variable
selected_option.trace("w", update_custom_ai_text_label)

# Chat start(checks for your option, API KEY)
def start_chat_thread():
    global custom_ai_selected
    selected = selected_option.get()
    if selected == "Custom AI":
        custom_ai_selected = True
    else:
        custom_ai_selected = False
    # Create a thread for running the chat
    openai.api_key = input_box.get()
    if openai.api_key == "":
        print('WRONG API KEY')
        root.destroy()
        sys.exit(0)
    topic = topic_box2.get()
    if topic == '':
        print('TOPICS CANT BE EMPTY')
        root.destroy()
        sys.exit(0)
    chat_thread = Thread(target=start_chat)
    chat_thread.start()

button_image = Image.open("button_start.png")  # Replace with the path to your button image # Adjust the size as per your image requirements
button_photo = ImageTk.PhotoImage(button_image)

button_image2 = Image.open("button_close.png")  # Replace with the path to your button image # Adjust the size as per your image requirements
button_photo2 = ImageTk.PhotoImage(button_image2)

# Create the start button
start_button = tk.Button(button_frame, image=button_photo, borderwidth=0, command=start_chat_thread)
start_button.pack(side="left", padx=10, pady=5)

# Create the close button
close_button = tk.Button(button_frame, image=button_photo2, borderwidth=0, command=close_window)
close_button.pack(side="left", padx=10, pady=5)

# Center the frame within the window horizontally
button_frame.place(relx=0.49, rely=0.95, anchor="s")
canvas_frame.place(relx=0.4, rely=-0.08, anchor="n")

# Hide the frame background
style.configure("Transparent.TFrame", background=root["bg"])

root.protocol("WM_DELETE_WINDOW", close_window)

# Start the tkinter event loop
root.mainloop()

if __name__ == "__main__":
    start_chat()