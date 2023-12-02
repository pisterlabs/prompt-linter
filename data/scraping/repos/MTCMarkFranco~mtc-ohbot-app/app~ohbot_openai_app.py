import os
import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI
from dotenv import load_dotenv
import time
import requests
import cv2
import dlib
import math
import threading
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import gc
import numpy as np

load_dotenv(dotenv_path=".\\ENV\\local.env")

# Globals
messages = []
engageWithPerson = False
person_looking_at_history = []
start_time = None
captureDevice = None
detector = None
predictor = None
interact_thread = threading.Thread()

# Set up OpenAI API credentials and client object
oAIClient = AzureOpenAI(
    api_version="2023-07-01-preview",
    azure_endpoint=os.getenv("OPENAI_ENDPOINT")
)

# Set up Azure Speech-to-Text and Text-to-Speech credentials
speech_key = os.getenv("SPEECH_KEY")
service_region = os.getenv("REGION")
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
# Set up Azure Text-to-Speech language 
speech_config.speech_synthesis_language = os.getenv("RECOGNITION_LANGUAGE")
# Set up Azure Speech-to-Text language recognition
speech_config.speech_recognition_language = os.getenv("RECOGNITION_LANGUAGE")

# Set up the voice configuration
speech_config.speech_synthesis_voice_name = "en-CA-ClaraNeural"
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

# start a new conversation context
def start_new_conversation():
    global messages
     
    messages.clear()
    gc.collect()
    
    # setup the messages and system prompt List defaults
    Instruction =   "You are a friendly person, looking to have friendly dialogue with whoever you speak with. " \
                    "You will answer questionsand ask questions. " \
                    "You will not be rude or mean. " \
                    "You will not use profanity. " \
                    "You will not be racist or sexist. " \
                    "Please be friendly and have a good conversation. " \
                    "add some humour to your responses including some laughing text in the form of 'hahaha' in the form of text, no emojis. " \
                    "Only return responses that can be converted safely to UTF-8 format. " \
                    "Your name is Zira." \
                    "if not provided, you should ask for their name before giving a response. " \
                    "start off every response with the person's name. " \
                    "if you didn't understand the question or were given a partial sentence, response with 'I didn't quite get that, please try again.' " \
                    "try to make small talk if there isn't a direct question being asked" \
                    "Your first response back to the user should be 'Hi There, what is your name?' " \
                    "if you do not have access to real-time data , try to find the information being asked and as a last resort " \
                    "say that you do not have access to that information at this time." \

    messages=[
                {
                    "role": "system", 
                    "content": Instruction
                }
        ]
    print("Starting a new conversation")

# Define the speech-to-text function
def speech_to_text():
    # Set up the audio configuration
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)

    # Create a speech recognizer and start the recognition
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    print("Your turn to speak...")

    result = speech_recognizer.recognize_once_async().get()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        return ""
    elif result.reason == speechsdk.ResultReason.Canceled:
        return ""

# Define the Azure OpenAI language generation function
def generate_text(prompt):
    global messages

    try:

        messages.insert(messages.__len__(), 
                        {
                            "role": "user", 
                            "content": prompt
                        })
    
    
        completion = oAIClient.chat.completions.create(
            model=os.getenv("MODEL"),   
            messages=messages,
        )
        
        messages.insert(messages.__len__(), 
                        {
                            "role": "assistant", 
                            "content": completion.choices[0].message.content
                        })
        
        return completion.choices[0].message.content

    except Exception:
        return "Sorry, I ran into a problem, can you repeat that?"

# Define the text-to-speech function
def send_message_to_ohbot_service(text):
    try:
        url = "http://127.0.0.1:8000"
        headers = {"Content-Type": "text/html"}
        response = requests.post(url, headers=headers, data=text)
        response.raise_for_status()
        return True
    except Exception as ex:
        print(f"Error sending to Ohbot service: {ex}")
        return False
    
# Define the text-to-speech function
def send_gesture_to_ohbot_service(gesture):
    try:
        url = "http://127.0.0.1:8000/gesture"
        headers = {"Content-Type": "text/html"}
        response = requests.post(url, headers=headers, json=gesture)
        response.raise_for_status()
        return True
    except Exception as ex:
        print(f"Error sending to Ohbot service: {ex}")
        return False
 
def interact():
    
    while True:
        global start_time
        
        try:
            user_input = ""
                    
            lock = threading.Lock()
            with lock:
                if engageWithPerson == False:
                    pass 
                    time.sleep(1)
                    continue
            
            if len(messages) == 1:
                user_input = "Introduce yourself and ask me for my name."
            else:
                user_input = speech_to_text()
            
            if user_input != "":
                print(f"You said: {user_input}")

                prompt = f"{user_input}"
                response = generate_text(prompt)
                print(f"AI said: {response}")

                gestureBlink = {
                    "gesture": "blink",
                    "velocity": 0.01
                }
                
                send_gesture_to_ohbot_service(gestureBlink)
                send_message_to_ohbot_service(response)
                
            else:
                if start_time is None:
                    start_time = time.time()
                    print("Starting timer")    
                
                if time.time() - start_time >= 20:
                    start_new_conversation()
                    start_time = time.time()
                
                print("new conversion in: " + str(20 - (time.time() - start_time)))    
                time.sleep(1)

        except Exception as e:
            print(f"An error occurred: {e}")

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = math.dist(eye[1], eye[5])
    B = math.dist(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = math.dist(eye[0], eye[3])

    # compute the eye aspect ratio
    try:
        ear = (A + B) / (2.0 * C)
    except ZeroDivisionError:
        ear = 0

    # return the eye aspect ratio
    return ear

def initalize_face_Detection():
    
    global captureDevice, detector, predictor
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("recognition_models\\shape_predictor_68_face_landmarks.dat")
    captureDevice = cv2.VideoCapture(0)   
    
    return

def is_person_looking_at():
    
    global captureDevice
    global detctor
    global predictor
    global person_looking_at_history
    
    try:
        lookingAtCamera = False
        EYE_AR_THRESH = 0.15
        X = 5
        Y = 5
            
        # Capture frame-by-frame
        ret, frame = captureDevice.read()
  
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Define range of color To Remove Ceiling lighting (May need to be modified based on Hue of Light in your Center)
        lower_color = np.array([243, 243, 243])
        upper_color = np.array([246, 247, 243])

        # Create a mask for the pixels within the color range
        mask = cv2.inRange(rgb_frame, lower_color, upper_color)

        # Change these pixels to black
        rgb_frame[mask != 0] = [0, 0, 0]

        # Convert the image back to BGR
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        # Perform face detection
        faces = detector(frame, 0)
        
        if len(faces) > 0:
            # Determine the facial landmarks for the face region
            x1 = faces[0].left()  # left point
            y1 = faces[0].top()  # top point
            x2 = faces[0].right()  # right point
            y2 = faces[0].bottom()  # bottom point
            
            # Calculate the center of the face and normalize the coordinatesfor Ohbot
            X = math.ceil(100 - (int((x1 + x2) / 2) * 100) / 640) / 100
            Y = math.ceil(100 - (int((y1 + y2) / 2) * 100) / 480) / 100
            # print(f"X: {X} , Y: {Y}")

            # Create landmark object
            landmarks = predictor(image=frame, box=faces[0])
            
            # Initialize lists to hold eye coordinates
            left_eye = []
            right_eye = []

            # Loop through all the points
            for n in range(36, 42):  # Loop for left eye
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                left_eye.append((x, y))

            for n in range(42, 48):  # Loop for right eye
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                right_eye.append((x, y))

            # Calculate the Eye Aspect Ratio for both eyes
            leftEAR = eye_aspect_ratio(left_eye)
            rightEAR = eye_aspect_ratio(right_eye)

            # Average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0
                     
            if ear >= EYE_AR_THRESH:
                lookingAtCamera = True
            else:
                lookingAtCamera = False
        else:
            lookingAtCamera = False
        
        # Logic here to determine if the person is looking at the Ohbot
        if lookingAtCamera:
            return True, X, Y
        else:
            return False, X, Y

    except Exception as e:
        print(f"An error occurred: {e}")
        return False, 0.5, 0.5  
           
def mute_microphone():
    devices = AudioUtilities.GetMicrophone()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volume.SetMute(1, None)

def unmute_microphone():
    devices = AudioUtilities.GetMicrophone()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volume.SetMute(0, None)
    
              
#  *****************************************************
#  ************** MAIN PROGRAM FLOW ********************
#  *****************************************************


# Initialize a new conversation
start_new_conversation()

# Initialize Cature device, face detector and facial landmark predictor
initalize_face_Detection()

while True:
    
         
    # Check if there is a person looking at the camera(Ohbot) and get the coordinates of the person's face    
    isLookingAtMe, X,Y = is_person_looking_at()

              
    # if the person is looking at the camera, conversate, new or existing....  
    if isLookingAtMe:
                
        # enable the microphone
        # unmute_microphone()
        time.sleep(0.1)
        engageWithPerson = True
        
        # Check if the thread is defined and if it's still running
        if not ('interact_thread' in locals() and interact_thread.is_alive()):
           print('Starting a new interact thread...')
           interact_thread = threading.Thread(target=interact, daemon=True)
           interact_thread.start() 
        # fill head tracking object
        gestureLookAt = {
            "gesture": "lookAt",
            "head_coordinates": {
                "X": X ,
                "Y": Y 
            },
            "eye_coordinates": {
                "X": X ,
                "Y": Y
            },
            "velocity": 0.01
        }
        send_gesture_to_ohbot_service(gestureLookAt)   
        
    else:
        
        # remove backgound conversations if no one is actually talking to the Ohbot
        engageWithPerson = False
        # mute_microphone()
        time.sleep(0.1)
           
    time.sleep(0.3)