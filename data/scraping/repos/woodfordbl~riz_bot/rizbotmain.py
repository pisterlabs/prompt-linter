import time
import openai
import requests
import os
import playsound
import speech_recognition as sr
import random
 
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options

from bs4 import BeautifulSoup as bs
from random import choice
from gtts import gTTS
from pyfirmata import Arduino, SERVO,util

# Initialize the recognizer
r = sr.Recognizer()

import geckodriver_autoinstaller
geckodriver_autoinstaller.install()

#Chat GPT API Key
openai.api_key = "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"

#Load Aurduino Info
#port = "/dev/tty.usbmodem142301"
#make sure pins are PWM
##arm1Pin = 9
#arm2Pin = 10 
#basePin = 11 

#board = Arduino(port)
#board.digital[arm2Pin].mode =SERVO
#board.digital[arm1Pin].mode =SERVO
#board.digital[basePin].mode =SERVO


# Servo Commands
"""
def rotateservo(pin, angle):
    board.digital[pin].write(angle)
    #print('turned')
    time.sleep(0.015) #adds a small delay between each servo movement 

def ArmSlam():
    for i in range(0,180):
        rotateservo(arm1Pin, i)
    for i in range(180,0,-1):
        rotateservo(arm1Pin, i)
    for i in range(0,15):
        rotateservo(arm1Pin, i)
    for i in range(15,0,-1):
        rotateservo(arm1Pin, i)
"""
def randomMovement():
    #use this function in a for loop that keeps checking if pickup line is still being spoken
    #returns how much each servo rotated in degrees (0-360)

    #chooses random upper range of angle rotation between 10-180 in steps of 5
    randPos1 = random.randrange(0, 180, 10) 
    randPos2 = random.randrange(0, 180, 10) 
    #randPosB = random.randrange(0, 180, 10) 
    maxAngle = 180
    #print(randPos1, randPos2)
    #moves servo by one degree each iteration
    for i in range(0,maxAngle):
        if i < randPos1:
            rotateservo(arm1Pin,i)
    for i in range(0,maxAngle):
        if i < randPos2:
            rotateservo(arm2Pin,i)
    # for i in range(0,maxAngle):
    #     if i < randPosB:
    #         rotateservo(basePin,i)

    for i in range(randPos1,0,-1):
        rotateservo(arm1Pin, i)
    for i in range(randPos2,0,-1):
        rotateservo(arm1Pin, i)
    # for i in range(randPosB,0,-1):
    #     rotateservo(arm2Pin, i)
# Webdriver Commands
def proxy_generator():
  response = requests.get("https://sslproxies.org/")
  soup = bs(response.content, 'html5lib')
  proxy = choice(list(map(lambda x:x.text, soup.findAll('td')[::8])))
  return proxy

# Riz Commands
def about_search(info):
    
    query = "linkedin profile " + info + " site:linkedin.com/in/"
    #print("Query:  " + query)
   
    # navigate to the Google homepage
    driver.get("https://www.google.com/")


    #identify search box
    search_box = driver.find_element("name", "q")

    #send querry
    search_box.send_keys(str(query))
    search_box.send_keys(Keys.ENTER)

    time.sleep(3)

    driver.find_element(By.CLASS_NAME, "yuRUbf").click()
   
    
    #Pull info from the About Section of the profile 
    time.sleep(2)
    
    try: 
        about = driver.find_element(By.XPATH, "/html/body/main/section[1]/div/section/section[2]/div/p")   
        about_text = about.text
    except:
        about_text = ""

    try:
        experience_section = driver.find_element(By.XPATH,"/html/body/main/section[1]/div/section/section[4]/div/ul")
        expereience_text = experience_section.text
    except: 
        expereience_text = ""

    link_profile = str(about_text) + "\n" + str(expereience_text)

    if link_profile:
        print("\nProfile found")
    else:
        print("No profile found")

    return(link_profile)
    
def riz_generator(about_text):
    
    max_tokens = 1024

    model_engine = "text-davinci-003"
    prompt = "I will provide a linked in profile description that contains the job history of an individual and occasionally a short bio that they write about themselves. Using a single aspect such as a job, a hobby, or accomplishment, create a pick-up line that is funny and integrates the aforementioned aspect. Return only a single pickup line in quotation marks. \n Profile Description: " + str(about_text)

    riz_line = openai.Completion.create(
    engine=model_engine,
    prompt=prompt,
    max_tokens=max_tokens,
    temperature=0.4,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return(riz_line.choices[0].text)

def text2speach(speach_text):
    # Language in which you want to convert
    language = 'en'
    #for accent
    SouthAfrica = 'com.au'

    print("\nRizz bot:  " + speach_text) 
    myobj = gTTS(text=speach_text, lang=language, slow=False, tld=SouthAfrica)
    
    # Saving the converted audio in a mp3 file named
    filename = "abc.mp3" 
    myobj.save(filename)
    #randomMovement()
    playsound.playsound(filename)
    os.remove(filename)
    return

def riz(input_string):
    start = input_string.find('"')
    end = input_string.find('"', start + 1)
    return input_string[start + 1:end]

def get_text(text):
    words = text.split() 
    for i in words:
        for i, item in enumerate(words):
            if item == "me":
                try:
                    for y in range(3):
                        new_string = " ".join(words[i + 2:i + 5])
                        return(new_string)
                except:
                    return(words)     

def text2riz(input):
    text2speach("Looking for a way to riz them now")
    text = about_search(input)
    pickupline = riz(riz_generator(text))
    #text2speach(pickupline)
    return(pickupline)

def speach2text():
    print("starting to listen")
    while(1):   
        
        # Exception handling to handle
        # exceptions at the runtime
        try:
            
            # use the microphone as source for input.
            with sr.Microphone() as source2:
                
                # wait for a second to let the recognizer
                # adjust the energy threshold based on
                # the surrounding noise level
                r.adjust_for_ambient_noise(source2, duration=0.2)
                
                #listens for the user's input
                audio2 = r.listen(source2)
                
                # Using google to recognize audio
                recieved_text = r.recognize_google(audio2)
                recieved_text = recieved_text.lower()
                print(recieved_text)
    
                if "hey robot" in recieved_text:
                    text2speach("Who can I help you riz?")
                    audio_information = r.listen(source2)
                    time.sleep(1.2)
                    info_text = r.recognize_google(audio_information)
                    if "help me" in info_text:
                        name_info = get_text(info_text)
                        text2riz(name_info)
                        playsound.playsound("rizzsong.mp3")
                    else:
                        text2speach("I didnt get that sorry")

        except sr.RequestError as e:
            print("Could not request results")

        except sr.UnknownValueError:
            print("unknown error occurred")

# Start FireFox
proxy = proxy_generator()
opts = Options()
opts.headless = False
opts.add_argument('--proxy-server=%s' % proxy)
driver = webdriver.Firefox(options=opts)
driver.get("https://www.google.com/")
input("\nPress Enter To Continue")

while True:
    #For Write in uncomment the following
    riz_line = text2riz(input("\nRizz bot: Who should I rizz:  "))
    print(riz_line)
    text2speach(riz_line)

    playsound.playsound("rizzsong.mp3")

#speach2text()