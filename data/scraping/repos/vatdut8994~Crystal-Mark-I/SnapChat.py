from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium import webdriver
import pandas as pd
import numpy as np
import pyperclip
import platform
import openai
import time
import os

# C:\\Users\\vatdu\\AppData\\Local\\Google\\Chrome\\User Data
with open("pwd.txt", 'r') as pwd:
    folder_location = pwd.read()
    
chromefile = open(f"{folder_location}database/chromepath.txt", 'r')
chromepath = chromefile.read()
chromefile.close()

try:
    options = webdriver.ChromeOptions()
    options.add_argument(f"user-data-dir={chromepath}")
    driver = webdriver.Chrome(executable_path="D:\Crystal3.0-20221010T223846Z-001\Crystal3.0\chromedrivers\chromedriver.exe", chrome_options=options)
except:
    print("FAILED TO FIND SPECIFIED CHROME PATH")
    chromepath = input("Enter the corrected chrome path: ")
    with open('chromepath.txt', 'w') as chromefile:
        chromefile.write(chromepath)
    options = webdriver.ChromeOptions()
    options.add_argument(f"user-data-dir={chromepath}")
    driver = webdriver.Chrome(executable_path="D:\Crystal3.0-20221010T223846Z-001\Crystal3.0\chromedrivers\chromedriver.exe", chrome_options=options)

months = ["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"]
openai.api_key = "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"
driver.get("https://web.snapchat.com/")

def get_chat():
    color = None
    # Name span class: mYSR9

    wait = WebDriverWait(driver, 600)
    my_chat = []
    reciever_chat = []
    final_name = ""
    person = "none"

    # x_arg = '//span[@class="ogn1z"]'
    names = wait.until(EC.presence_of_all_elements_located((
        By.CLASS_NAME, "FiLwP")))
    statuses = wait.until(EC.presence_of_all_elements_located((
        By.CLASS_NAME, "GQKvA")))
    for name, status in zip(names, statuses):
        if "New Chat" in status.text:
            # print(name.text)
            final_name = name.text
            person = name
            name.click()
            time.sleep(2)
            chats = wait.until(EC.presence_of_all_elements_located((
                By.CLASS_NAME, "T1yt2")))
            chat_list = []
            for chat in chats:
                chat_list.append(chat.text)
            for chat_element in chat_list:
                me = []
                other = []
                for message in chat_element.split('\n'):
                    if message == "ME":
                        # print("My turn", end=": \n")
                        color = 'red'
                        dontadd = True
                    elif message == name.text.upper():
                        color = 'blue'
                        # print("Your turn", end=": ")
                        dontadd = True
                    for month in months:
                        try:
                            if message.split()[0] == month and message.split()[1].isdigit():
                                # print("Oooof its a date")
                                color = None
                                dontadd == True
                            else:
                                pass
                        except:
                            pass
                    if color == 'red' and dontadd == False:
                        me.append(message.replace('�', "'"))
                        # print(message, end=', \n')
                    elif color == "blue" and dontadd == False:
                        other.append(message.replace('�', "'"))
                        # print(message, end=', \n')
                    dontadd = False
                if me != []:
                    my_chat.append('. '.join(me))
                if other != []:
                    reciever_chat.append('. '.join(other))
    
    if len(my_chat) > len(reciever_chat):
        my_chat = my_chat[(len(my_chat)-len(reciever_chat))::]
    elif len(my_chat) < len(reciever_chat):
        reciever_chat = reciever_chat[(len(reciever_chat)-len(my_chat))::]

    my_chat.insert(0, "ME")
    reciever_chat.insert(0, "RECIEVER")
    return person, final_name, {"ME":my_chat, "RECIEVER":reciever_chat}

def format_data(other_name):
    # with open("chatsnap.csv", 'r', encoding='utf-8') as filer:
    #     readed = filer.read()
    # readed = readed.replace('�', "'")
    # print(readed)
    # with open("chatsnap1.csv", 'w', encoding="utf-8") as filew:
    #     filew.write(readed)
    try:
        data = pd.read_csv(f"{folder_location}database/chatsnap.csv", delimiter="|", encoding='utf-8')
    except:
        with open(f"{folder_location}database/chatsnap.csv", 'w') as file:
            file.write("ME|RECIEVER\n")
        data = pd.read_csv(f"{folder_location}database/chatsnap.csv", delimiter="|", encoding='utf-8')
        
    my_chat = data["ME"]
    other_chat = data["RECIEVER"]
    full = ""
    for me, other in zip(my_chat, other_chat):
        full = full + f"Me: {me}\n{other_name.title()}: {other}\n"
    return full + "Me:"

def paste_keys(el, text):
    os.environ['CLIPBOARD'] = text
    if platform.uname()[0] == 'Darwin':
        el.send_keys(Keys.COMMAND, 'v')
    else:
        el.send_keys(Keys.CONTROL, 'v')

def generate_reponse(person, previous, name):
    if person != "none":
        response = openai.Completion.create(
            model="text-davinci-003", temperature=1, max_tokens=2000, prompt= f"You are embedded in a SnapChat chatbot and you will talk to {name} now. Try to act like an actual human, and like me, Vatsal Dutt.\n"+previous,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0.6,
            stop=[f" {name}:", " Me:"]
        )
        response = response.choices[0].text
        print(response)
        wait = WebDriverWait(driver, 600)
        text_box = wait.until(EC.presence_of_element_located((
            By.CLASS_NAME, "euyIb")))
        pyperclip.copy(response)
        paste_keys(text_box, response)
        text_box.send_keys(Keys.ENTER)
        person.click()


while True:
    person, name, all_chat = get_chat()
    data = pd.DataFrame(all_chat)
    if len(all_chat["ME"]) > 1 and len(all_chat["RECIEVER"]) > 1:
        np.savetxt(f'{folder_location}database/chatsnap.csv', data.to_numpy(),fmt='%s', delimiter='|', encoding="utf-8")
    previous_chat = format_data(name)
    generate_reponse(person, previous_chat, name)
    time.sleep(10)
