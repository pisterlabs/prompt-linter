import openai
import os
import json
import requests
import webbrowser
import random 
import docx2txt
import PySimpleGUI as sg
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
from pydub import AudioSegment
from dotenv import load_dotenv
from googlesearch import search
import re
from bs4 import BeautifulSoup

load_dotenv()
terminalOutputs = []



def extract_quoted_content(rawQuery):
    pattern = r'"([^"]*)"'
    matches = re.findall(pattern, rawQuery)
    return matches

def GPT4():
    
    if values["-SEARCH-"] == True:
        updatedMessage = "Use this data: " + textResults
        messages.append( {"role": "user", "content": updatedMessage} )
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            top_p=topP,
            temperature=temp,                                                   
            frequency_penalty=0.0,                                              
            presence_penalty=0.0 )
        
    else:
        messages.append( {"role": "user", "content": message} )
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            top_p=topP,
            temperature=temp,                                                   
            frequency_penalty=0.0,                                              
            presence_penalty=0.0 )
    
    reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": reply})
    print("\n" + reply + "\n")
    conversation.append(reply)

def GPT4WithGoogle():
    googleSearch = "whats a good google search to answer this question?" + message
    messages.append( {"role": "user", "content": googleSearch} )
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        top_p=topP,
        temperature=temp,                                                   
        frequency_penalty=0.0,                                              
        presence_penalty=0.0 )
    
    rawQuery = response["choices"][0]["message"]["content"]
    conversation.append(rawQuery)

    goodQueryList = extract_quoted_content(rawQuery)
    goodQueryString = " ".join(goodQueryList)
    query = goodQueryString
    print(query)
    search_results = []

    for results in search(query, tld="co.in", num=1, stop=1, pause=2):
        search_results.append(results)
        conversation.append(results)
    
    page = requests.get(search_results[0])
    soup = BeautifulSoup(page.content, 'html.parser')
    
    textResults = soup.get_text()
    textResults = textResults.strip()
    textResults = textResults.replace("\n", " ")
    textResults = textResults[:8000]
    conversation.append(textResults)
    return textResults    

    
def generate_dall_e_image(prompt):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {openai.api_key}'
    }
    data = {
        'model': 'image-alpha-001',
        'prompt': f'{prompt}',
        'num_images': 1,
        'size': '1024x1024'
    }
    url = 'https://api.openai.com/v1/images/generations'

    # Send the request to the DALL-E API
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response_json = response.json()

    # Check if the response was successful
    if response.status_code == 200 and 'data' in response_json and len(response_json['data']) > 0:
        return response_json['data'][0]['url']
    else:
        return None

def whisper_transcript():
        
        strippedLocation = docURL
        audioFile = open(strippedLocation, "rb")
        transcript = openai.Audio.transcribe("whisper-1", audioFile)
        transcript = str(transcript)
        
        #remove the {} from the start and end of the transcript
        finalTranscript = transcript[12:-1]
        message = "If possible, use this as a primary source for any further questions:" + finalTranscript

def browse_textfiles():
    sourceLocation = message.split("R>")[1].strip()
    with open(sourceLocation) as file:
        txtContent = file.read().replace('\n',' ')
    message = "If possible, use this as a primary source for any further questions:" + txtContent

def siriGPT():
    freq = 44100
    duration = recordingDurationSeconds
    recording = sd.rec(int(duration * freq), samplerate=freq, channels=2)
    duck_img.update(filename=f'{folderlocation}\duckspeak.png')
    sd.wait()
    duck_img.update(filename=f'{folderlocation}\duckwaddle1.png')
    write("output.wav", freq, recording)
    os.open("output.wav", os.O_RDONLY)
    sound = AudioSegment.from_wav('output.wav')
    sound.export('finalOutput.mp3', format='mp3')
    print(sound)

##extra encouragement;)
messages = [
    {"role": "system", "content": "Hello, what can I do for you today?"},
    {"role": "user", "content": "You are a helpful assistant."},
    {"role": "system", "content": "Thank you, I try my best."}
]
conversation = []
system_msg = "You are an AI."
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

#customize the GUI
sg.theme("DarkGreen2")
fontT = ("Arial", 20)
fontB = ("Arial", 15)
fontS = ("Arial", 10)
menu = [["Settings", ["Change Temperature", "Change Top-P", "Change Recording Duration", "Change API Key"]], ["Debug", ["Show Terminal"]]]
title = sg.Text("To manually import a .txt file begin with R> \nTo manually import an audio file begin with L>", font=fontT)
folderlocation = os.path.dirname(os.path.realpath(__file__))
duck_img = sg.Image(f'{folderlocation}/duck.png', size=(130,130), enable_events=True, key="-DUCK-")
mic_button = sg.Button("Audio Input", key="-RECORD-")
send_button = sg.Button("Send")
clear_button = sg.Button("Clear")
quit_button = sg.Button("Quit")
internetsearch = sg.Checkbox("Search Internet", default=False, key="-SEARCH-")
#tempSlider = sg.Text("Temperature:", font=fontS), sg.Slider(range=(0, 10), default_value=5, orientation='h', size=(34, 10), font=fontS, key="-SLIDER1-")
#topPSlider = sg.Text("Top-P:", font=fontS), sg.Slider(range=(0, 10), default_value=5, orientation='h', size=(34, 10), font=fontS, key="-SLIDER2-")
#recordingDuration = sg.Text("Recording Duration:", font=fontS), sg.Slider(range=(0, 10), default_value=5, orientation='h', size=(34, 10), font=fontS, key="-SLIDER3-")
mainInput = sg.Multiline("", size=(100,5), key="-INPUT-", font=fontB)
secondaryInput = sg.Input(enable_events=True, key='-IN-',font=fontS, expand_x=True), sg.FileBrowse()
mainOutput = sg.Multiline("", size=(100,25), key="-OUTPUT-", font=fontB)
TempAsk = 5
TopPAsk = 5
waddle_position = 0
results = ""
textResults = ""


#define the layout of the GUI
layout = [[sg.Menu(menu)],
        [title, sg.Column([[duck_img]], justification='right')],
        #[tempSlider, topPSlider, recordingDuration]
        [mainInput],
        [secondaryInput],
        [internetsearch, send_button, clear_button, quit_button, sg.Column([[mic_button]], justification='right')],
        [mainOutput]
    ]



# Create the GUI window
window = sg.Window("GPT-4", layout, size=(700, 700))

# Start the event loop
while True:
    
    event, values = window.read()
    if event == "Quit" or event == sg.WIN_CLOSED:
        break
    
    while True: 
        if event == "Change API Key":
            APIASK = sg.popup_get_text("Please enter your API Key", title="API Key")
            print("Key = ", APIASK)
            if APIASK != None:
                OPENAI_API_KEY = APIASK
            break
        elif event == "-RECORD-":
            ## RECORD AUDIO and make it into a temporary .mp3 file
            siriGPT()
            break
        elif event == "Show Terminal":
            TerminalContent = '\n'.join(terminalOutputs)
            TerminalPopup = sg.popup_scrolled(TerminalContent + "this doesnt work yet, please help", size=(100, 100), title="Terminal", font=fontS)
            print("Terminal Opened")
            break
        elif event == "Change Temperature":
            TempAsk = sg.popup_get_text("Enter Temperature value between 1-10", title="Temperature")
            if TempAsk == None:
                TempAsk = 5
            print("Temperature Changed")
            break
        elif event == "Change Top-P":
            TopPAsk = sg.popup_get_text("Enter Top-P value between 1-10", title="Top-P")
            if TopPAsk == None:
                TopPAsk = 5
            print("Top-P Changed")
            break
        elif event == "Change Recording Duration":
            RecordingDurationAsk = sg.popup_get_text("Enter Recording Duration ", title="Recording Duration")
            if RecordingDurationAsk == None:
                RecordingDurationAsk = 5
            recordingDurationSeconds = RecordingDurationAsk
            print("Recording Duration Changed")
            break
        else:
            break
            
    
    docURL = values["-IN-"]
    if values["-IN-"] != None:
        if ".mp3" in docURL:
            message = docURL
            whisper_transcript()
        elif ".txt" in docURL:
            message = ("R>" + docURL)
        elif ".docx" in docURL:
            convFile = docx2txt.process(docURL)
            txtContentCV = convFile.replace('\n',' ')
            message = "If possible, use this as a primary source for any further questions:" + txtContentCV
            
    else:
        messagePreRep = values["-INPUT-"]
        message = messagePreRep.replace('\n',' ')
        conversation.append(message)
        
    if event == "Clear":
        window["-INPUT-"].update("")
        window["-OUTPUT-"].update("")
        conversation = []
        messages = []
        messages.append( {"role": "user", "content": "Hello, what can I do for you today?"} )
        messages.append( {"role": "system", "content": "You are a helpful assistant."} )
        messages.append( {"role": "user", "content": "Thank you, I try my best."} )
        continue
    
    
    temp = (float(TempAsk) * 0.1)
    topP = (float(TopPAsk) * 0.1)

    
    
    if event == "Send":
        if values["-SEARCH-"] == True:
            GPT4WithGoogle()
            conversationString = "\n \n".join(conversation)
            window["-OUTPUT-"].update(conversationString)
            window["-IN-"].update("")
            window["-INPUT-"].update("")
        else:
            GPT4()
            conversationString = "\n \n".join(conversation)
            window["-OUTPUT-"].update(conversationString)
            window["-IN-"].update("")
            window["-INPUT-"].update("")
    
    

    #OPEN TEXT FILES
    if "R>" in message:
        browse_textfiles()
    #TRANSCRIBE AUDIO INTO PROMPT
    ##if "L>" in message:
        ##whisper_transcript()


    
    """ remember to fix this later!!!
    #Create photo with DALL-E
    if "~" in reply:
        split_prompt = reply.split("~")[1].strip()
        image_url = generate_dall_e_image(split_prompt)
        print(image_url)
        #Open photo automatically
        if image_url != None:
            webbrowser.open(image_url)
    """
            
