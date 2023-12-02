import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from os.path import basename
import openai
import json
import re
import pyttsx3
from word2number import w2n
import pyaudio
import wave
import speech_recognition as sr
import sys 

gpt_stuff = r""
sys.path.insert(0, gpt_stuff)

from gpt_commands import list_of_commands

email_list = r""
sys.path.insert(0, email_list)

from email_names import known_emails


converter = pyttsx3.init()
converter.setProperty('rate', 150)
converter.setProperty('volume', 0.85)

r = sr.Recognizer()

FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000


def pierre_speak(phrase):
    converter.say(phrase)
    converter.runAndWait()
    
    

def multiple_attachment_listner(seconds):
    p = pyaudio.PyAudio()
    
    stream = p.open(
       format=FORMAT,
       channels=CHANNELS,
       rate=RATE,
       input=True,
       frames_per_buffer=FRAMES_PER_BUFFER
    )

    print("Pick a file number")

    frames = []
    for i in range(0, int(RATE / FRAMES_PER_BUFFER * seconds)):
        data = stream.read(FRAMES_PER_BUFFER)
        frames.append(data)

    print("....")

    stream.stop_stream()
    stream.close()
    p.terminate()


    wf = wave.open("attachment_number.wav", 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    
def audio_to_text():
    with sr.AudioFile("attachment_number.wav") as source:
        audio = r.record(source)

        try:
            raw_text = r.recognize_google(audio, show_all=True) #show all prevents error if no audio
            data = raw_text['alternative'][0]
            
            print(data)
            return (data)
            

            
        except TypeError as e:
            #For when no audio is recognized
            return {"transcript": ""}
    
    

    
def clean_command_params(raw_params):
    """Takes in a string of paramets from gpt and converts to python dict"""
    escape_cleaner = re.compile('(?<!\\\\)\'')
    
    #Remove new line characters from string
    new_text = []
    for char in raw_params:
        if char != "\n":
            new_text.append(char)

    command_parameters = "".join(new_text)

    #Remove escape backslashes from string
    p = re.compile('(?<!\\\\)\'')
    command_parameters = p.sub('\"', command_parameters)
    
    json_commands = json.loads(command_parameters) #Convert string to JSON
    print(command_parameters)
    
    
    #remove file string from file path
    if json_commands["file_name"] != "":
        if "file" == json_commands["file_name"].split(".")[-1]:
            json_commands["file_name"] = " ".join(json_commands["file_name"].split(".")[:-1])
            
        elif "file" == json_commands["file_name"].split()[-1]:
            json_commands["file_name"] = " ".join(json_commands["file_name"].split()[:-1])

        json_commands["file_name"] = json_commands["file_name"].replace("/", " ").strip()
    
    
        #remove folder string from file path
    if json_commands["file_path"] != "":
        if "folder" == json_commands["file_path"].split()[-1]:
            json_commands["file_path"] = " ".join(json_commands["file_path"].split()[:-1])

        json_commands["file_path"] = json_commands["file_path"].replace("/", " ").strip()
        
        
    return json_commands
        
        


def extract_email_command(email_command):
    """Convert the command into a dictionary of parameters using gpt"""
    
    
    #This is where I will send off commands to my own model once build and return the parameters in this function
    openai.api_key = ""

    gpt_email_prompt = list_of_commands["send_email_commands"][0] + email_command + list_of_commands["send_email_commands"][1]
    
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=gpt_email_prompt,
        max_tokens=700,
        temperature=0
    )

    text = response['choices'][0]['text'].lower()

    command_params = clean_command_params(text)
    
    #Converts email reciever name to actual address
    if command_params['to'].lower() in known_emails:
        command_params['to'] = known_emails[command_params['to']]
        
    
    return command_params




def recursive_folder_search(folder_path, folder_list, file_list):
    for item in os.listdir(folder_path):
        possible_folder_path = (folder_path + "\\" + item)
        
        if not os.path.isdir(possible_folder_path):
            file_list.append(possible_folder_path)
    
    
    #loop over all items in a folder
    for item in os.listdir(folder_path):
        if item == "fullstack redo" or item == "node_modules":
            #Prevents use searching extremely large files where we know the item isnt
            continue
        
        possible_folder_path = (folder_path + "\\" + item)
        
        #if an item is a folder open it and check its folders
        if os.path.isdir(possible_folder_path):
            folder_list.append(possible_folder_path)
            
            recursive_folder_search(possible_folder_path, folder_list, file_list)
            
    return


    
def verify_folder_path(path):
#     if path in global_paths:
#         #merge file path and name and check if it exists
#         #where I will have custom paths like desktop/homework
#         return
    
    #else:
    #Will only take main files like desktop (C:\Users\sbuca -> all folders in here)
    folder_path = ""

    for folder in [path, path.capitalize(), path.upper()]:
        folder = "~/" + folder
        folder_path = os.path.normpath(os.path.expanduser(folder))

        if os.path.exists(folder_path):
            break

        folder_path = ""

    #Will stop searching if the folder doesnt exist
    if folder_path == "":
        return None
        
    return folder_path
        
        

        
def find_file(folder_name, file_name):
    file_name = file_name.lower()
    matched_files = []
    
    folder_path = verify_folder_path(folder_name)
    
    if folder_path:
        #If the desired path if found recurse through every folder in it
        folder_list = []
        file_list = []
        recursive_folder_search(folder_path, folder_list, file_list)
        
        
        
        for file in file_list:
            split_file = file.split("\\")[-1].split(".")[0]

            if file_name in split_file.lower():
                print(file, split_file)
                matched_files.append(file)
#                 distance = lev.distance(Str1,Str2)
#                 ratio = lev.ratio(Str1,Str2)

        return matched_files if matched_files != [] else "No Files Found"       


def attachment_manager(all_files):
    """Handles all the files found given the name and folder
        manages when multiple files are found"""
    
    pierre_phrase = """I found {} files within that directory I will 
                        list them out now and you say the number of 
                        which one is correct""".format(len(all_files))
    
    pierre_speak(pierre_phrase)
    
    files_by_index = {}
    
    for i, file in enumerate(all_files):
        file_name = file.split("\\")[-1]
        file_name = file_name.replace("_", " ")
        
        files_by_index[str(i+1)] = file_name
        
        pierre_speak("{}, {}".format([i+1], file_name))
        
        print(files_by_index)
        
        
    file_attachment_number = ""
    while file_attachment_number == "":     
        multiple_attachment_listner(seconds=3)
        file_attachment_number = audio_to_text()["transcript"]

        if "repeat" in file_attachment_number.lower():
            #file_attachment_number = ""
            return attachment_manager(all_files)
        
        if file_attachment_number == "":
            continue
        
        if len(file_attachment_number) == 1: #Good to go
            return all_files[int(file_attachment_number)-1]

        else:
            print(file_attachment_number)
            for word in file_attachment_number.split():
                if word == "to" or word == "too":
                    word = "two"
                    
                try:
                    file_attachment_number = w2n.word_to_num(word)
                    return all_files[int(file_attachment_number)-1]

                except ValueError as e:
                    print(word)
                    continue
                    
        file_attachment_number = ""
            
            
def attachment_file_handler(folder_name, file_name):
    files = find_file(folder_name, file_name)
    
    if files == "No Files Found":
        #handle it
        return None
    
    if len(files) == 1:
        return files[0]
    
    
    return attachment_manager(files)



def add_attachment(msg, file_path):
    with open(file_path, "rb") as fil:
        part = MIMEApplication(
            fil.read(),
            Name=basename(file_path)
        )
    # After the file is closed
    part['Content-Disposition'] = 'attachment; filename="%s"' % basename(file_path)
    msg.attach(part)



def pacakge_email_data(sender, email_params):
    msg = MIMEMultipart()
    msg['From'] = sender    
    msg['To'] = email_params['to']    
    msg['Subject'] = email_params['subject']

    #Add body to email
    body = MIMEText(email_params['body'])
    msg.attach(body)
    
    #Add atachment to email -> Update in futrue to have multiple
    if email_params['file_path'] != "" and email_params['file_name'] != "":
        attachment_file_path = attachment_file_handler(email_params['file_path'], email_params['file_name'])
    
        if attachment_file_path:
            add_attachment(msg, attachment_file_path)
        
        else:
            return "Could Not Find File"
    
    return msg


def send_email(sender, sender_password, receiver, email_data):
    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    
    server.ehlo()
    
    server.login(sender, sender_password)

    server.sendmail(sender, receiver, email_data.as_string())

    server.close()
    
    return "Success"


def email_main(email_command, email_params=""):
    email_sender = ""
    email_app_password = ""
    
    email_params = extract_email_command(email_command)
    print(email_params)
    
    email_data = pacakge_email_data(email_sender, email_params)
    
    status = send_email(email_sender, email_app_password, email_params["to"], email_data)
    

if __name__ == "__main":
    email_main(email_command, email_params="")
