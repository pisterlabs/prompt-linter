"""
This program generates photos from random audio conversations and displays 
them on the screen. When run on command, it is an interesting exercise in the power of OpenAI.
When run in continuous mode, it is a creepy photo generator because it shows how good openAI is 
at understanding what you are saying.

There is a step that is now commented out that summarizes the transcript. The summary is 
errily accurate.

Basic flow:
    * record audio from the default microphone and then transcribe it using OpenAI
    * summarize the transcript and generate 4 pictures based on the summary
    * combine the four images into a single image
    * open the picture in a browser
    * optionally, delay for 60 seconds and repeat the process
    * images are stored in the history directory

The program can be run in two modes:
    
1/  python3 pyspeech.py
    This will display a menu and prompt you for a command. 
2/  python3 pyspeech.py -h
    For testing. Use command line arguments  

control-c to stop the program. When run in auto mode it will loop 10 times

For debug output, use the -d 2 argument. This will show the prompts and responses to/from OpenAI.

To run this you need to get an OpenAI API key and put it in a file called "creepy photo secret key".
OpenAI currently costs a few pennies to use. I've run this for an hour at a cost of $1.00. It was
well worth it.

ALSO NOTE: If you are not getting any audio, then you may not have given the program
permission to access your microphone. On OSX it took me some searching to figure this out.
https://superuser.com/questions/1441270/apps-dont-show-up-in-camera-and-microphone-privacy-settings-in-macbook
Until the Terminal app showed up in Settings / Privacy & Security / Microphone this program
just wont work. 
On the RPi I had to add my user to the "audio" group. I did this
with      usermod -a -G audio <username>

Based on the WhisperFrame project idea on Hackaday.
https://hackaday.com/2023/09/22/whisperframe-depicts-the-art-of-conversation/

Specific to Raspberry Pi:
    1. set up a virtual environment and activate it (to deactive use "deactivate")
        python3 -m venv .venv
        source .venv/bin/activate

        set your openai key
            https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety
                echo "export OPENAI_API_KEY='yourkey'" >> ~/.zshrc
                source ~/.zshrc
                echo $OPENAI_API_KEY  # to verify
                hmmm, we don't do this yet it works ... openai.api_key = os.environ["OPENAI_API_KEY"]
    
    2. install the following packages

        2a. for RPi version 3 install these
            sudo apt-get install portaudio19-dev
            sudo apt-get install libasound2-dev
            sudo apt-get install libatlas-base-dev
            sudo apt-get install libopenblas-dev
            # sudo apt-get install feh

        2b. on MacOS install these
            brew install portaudio
            brew update-reset   # if your brew isn't working try this
            xcode-select --install  # needed for pyaudio to install
            pip3 install sounddevice
            pip3 install soundfile
            pip3 install numpy

            Use finder and navigate to /Applications/Python 3.12
                  Then doublelick on "Install Certificates.command"

                    
    3. install the following python packages    
        pip install openai # --upgrade
        pip install pillow
        pip install pyaudio
        pip install RPi.GPIO

    Note that when run you will see 10 or so lines of errors about sockets and JACKD and whatnot.
    Don't worry, it is still working. If you know how to fix this, please let me know.

    Also note that errors from the audio subsystem are ignored in recordAudioFromMicrophone(). If 
    you are having some real audio issue, you might change the error handler to print the errors.
    
Author: Jim Schrempp 2023 

v 0.5 Initial version
v 0.6 2023-11-12 inverted Go Button logic so it is active low (pulled to ground)
v 0.7 updated to python 3.12 and openAI 1.0.0 (wow that was a pain)
      BE SURE to read updated install instructions above
"""

# import common libraries
import platform
import argparse
import logging
import urllib.request
import time
import shutil
import re
import os
import json 
import openai
from openai import OpenAI

client = OpenAI() 
from enum import IntEnum
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__) # parameter: -d 1
loggerTrace = logging.getLogger("Prompts") # parameter: -d 2

# import platform specific libraries
g_isMacOS = False
if (platform.system() == "Darwin"):
    g_isMacOS = True
else:
    print ("Not MacOS")

if g_isMacOS:
    import sounddevice
    import soundfile

else:
    # --------- set for Raspberry Pi -----------------------------------------
    import pyaudio
    import wave
    from ctypes import *
    import RPi.GPIO as GPIO
    import threading
    from queue import Queue


# Set the duration of each recording in seconds
duration = 120

# Set the number of times to loop when in auto mode
loopsMax = 10

# Prompt for abstraction
promptForAbstraction = "What is the most interesting concept in the following text \
    expressing the answer as a noun phrase, but not in a full sentence "

# image modifiers
imageModifiersArtist = [
                    "Picasso",
                    "Van Gogh",
                    "Monet",
                    "Dali",
                    "Escher",
                    "Rembrandt",
                    ]
imageModifiersMedium = [
                    "painting",
                    "watercolor",
                    "sketch",
                    "vivid color",
                    "photograph",
                    ]

# Define  constants for blinking the LED (onTime, offTime)
constBlinkFast = (0.1, 0.1)
constBlinkSlow = (0.5, 0.5)
constBlinkAudioCapture = (0.05, 0.05)
constBlink1 = (0.5, 0.2)
constBlink2 = (0.4, 0.2)
constBlink3 = (0.3, 0.2)
constBlink4 = (0.2, 0.2)

constBlinkStop = (-1, -1)
constBlinkDie = (-2, -2)

if not g_isMacOS:
    # --------- Raspberry Pi specific code -----------------------------------------
    logger.info("Setting up GPIO pins")

    g_LEDRed = 8
    g_goButton = 10

    # Set the pin numbering mode to BCM
    GPIO.setmode(GPIO.BOARD)

    # Set up pin g_LEDRed as an output
    GPIO.setup(g_LEDRed, GPIO.OUT, initial=GPIO.LOW)
    
    # Set up pin 10 as an input for the start button
    GPIO.setup(g_goButton, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    g_buttonPressed = GPIO.LOW

    # Define a function to blink the LED
    # This function is run on a thread
    # Communicate by putting a tuple of (onTime, offTime) in the qBlinkControl queue
    #
    def blink_led(q):
        # print("Starting LED thread") # why do I need to have this for the thread to work?
        logger.info("logging, Starting LED thread")

        # initialize the LED
        isBlinking = False
        GPIO.output(g_LEDRed, GPIO.LOW)

        while True:
            # Get the blink time from the queue
            try:
                blink_time = q.get_nowait()
            except:
                blink_time = None

            if blink_time is None:
                # no change
                pass
            elif blink_time[0] == -2:
                # die
                logger.info("LED thread dying")
                break
            elif blink_time[0] == -1:
                # stop blinking
                GPIO.output(g_LEDRed, GPIO.LOW)
                isBlinking = False
            else:
                onTime = blink_time[0]
                offTime = blink_time[1]
                isBlinking = True

            if isBlinking:
                # Turn the LED on
                GPIO.output(g_LEDRed, GPIO.HIGH)
                # Wait for blink_time seconds
                time.sleep(onTime)
                # Turn the LED off
                GPIO.output(g_LEDRed, GPIO.LOW)
                # Wait for blink_time seconds
                time.sleep(offTime)

    # Create a new thread to blink the LED
    logger.info("Creating LED thread")
    qBlinkControl = Queue()
    led_thread1 = threading.Thread(target=blink_led, args=(qBlinkControl,),daemon=True)
    led_thread1.start()


    # --------- end of Raspberry Pi specific code ----------------------------

# ----------------------
# change the blink rate
#   This routine isolates the RPi specific code
def changeBlinkRate(blinkRate):
    if not g_isMacOS:
        # running on RPi
        qBlinkControl.put(blinkRate)
    else:
        # not running on RPI so do nothing
        pass
                                
# ----------------------
# record duration seconds of audio from the default microphone to a file and return the sound file name
#
def recordAudioFromMicrophone():

    soundFileName = 'recording.wav'
    
    # delete file recording.wav if it exists
    try:
        os.remove(soundFileName)
    except:
        pass # do nothing   
    
    if g_isMacOS:
        # print the devices
        # print(sd.query_devices())  # in case you have trouble with the devices

        # Set the sample rate and number of channels for the recording
        sample_rate = int(sounddevice.query_devices(1)['default_samplerate'])
        channels = 1

        logger.debug('sample_rate: %d; channels: %d', sample_rate, channels)

        logger.info("Recording %d seconds...", duration)
        os.system('say "Recording."')
        # Record audio from the default microphone
        recording = sounddevice.rec(
            int(duration * sample_rate), 
            samplerate=sample_rate, 
            channels=channels
            )

        # Wait for the recording to finish
        sounddevice.wait()

        # Save the recording to a WAV file
        soundfile.write(soundFileName, recording, sample_rate)

        os.system('say "Thank you. I am now analyzing."')

    else:

        # RPi
        """
        recording = sounddevice.Stream(channels=1, samplerate=44100)
        recording.start()
        time.sleep(15)
        recording.stop()
        soundfile.write('test1.wav',recording, 44100)
        """

        # all this crap because the ALSA library can't police itself
        ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
        def py_error_handler(filename, line, function, err, fmt):
            pass #nothing to see here
        c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
        asound = cdll.LoadLibrary('libasound.so')
        # Set error handler
        asound.snd_lib_error_set_handler(c_error_handler)
        # Initialize PyAudio
        pa = pyaudio.PyAudio()
        # Reset to default error handler
        asound.snd_lib_error_set_handler(None)
        # now on with the show, sheesh

        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=44100,
            input=True,
            frames_per_buffer=1024
            ) #,input_device_index=2)

        wf = wave.open(soundFileName,"wb")
        wf.setnchannels(1)
        wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)

        # Write the audio data to the file
        for i in range(0, int(44100/1024*10)):

            # Get the audio data from the microphone
            data = stream.read(1024)

            # Write the audio data to the file
            wf.writeframes(data)

        # Close the microphone and the wave file
        stream.close()
        wf.close()

    return soundFileName

# ----------------------
# transcribe the audio file and return the transcript
#
def getTranscript(wavFileName):

    # transcribe the recording
    logger.info("Transcribing...")
    audio_file= open(wavFileName, "rb")
    responseTranscript = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file)

    # print the transcript object
    loggerTrace.debug("Transcript object: " + str(responseTranscript))

    transcript = responseTranscript.text 

    loggerTrace.debug("Transcript text: " + transcript)

    return transcript

# ----------------------
# summarize the transcript and return the summary
#
def getSummary(textInput):
    
    # summarize the transcript 
    logger.info("Summarizing...")

    responseSummary = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content" : 
        f"Please summarize the following text:\n{textInput}" }
    ])
    loggerTrace.debug("responseSummary: " + str(responseSummary))

    summary = responseSummary['choices'][0]['message']['content'].strip()
    
    logger.debug("Summary: " + summary)

    return summary

# ----------------------
# get keywords for the image generator and return the keywords
#
def getAbstractForImageGen(inputText):

    # extract the keywords from the summary

    logger.info("Extracting...")
    logger.debug("Prompt for abstraction: " + promptForAbstraction)    

    prompt = promptForAbstraction + "'''" + inputText + "'''"
    loggerTrace.debug ("prompt for extract: " + prompt)

    responseForImage = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": prompt}
    ])

    loggerTrace.debug("responseForImageGen: " + str(responseForImage))

    # extract the abstract from the response
    abstract = responseForImage['choices'][0]['message']['content'].strip() 
    
    # Clean up the response from OpenAI
    # delete text before the first double quote
    abstract = abstract[abstract.find("\"")+1:]
    # delete text before the first colon
    abstract = abstract[abstract.find(":")+1:]
    # eliminate phrases that are not useful for image generation
    badPhrases = ["the concept of", "in the supplied text is", "the most interesting concept"
                    "in the text is"]
    for phrase in badPhrases:
        # compilation step to escape the word for all cases
        compiled = re.compile(re.escape(phrase), re.IGNORECASE)
        res = compiled.sub(" ", abstract)
        abstract = str(res) 

    logger.info("Abstract: " + abstract)

    return abstract

# ----------------------
# get images and return the urls
#
def getImageURL(phrase):

    # use the keywords to generate an image

    prompt = "Generate a picture" 
    # add a modifier to the phrase
    # pick random modifiers
    import random
    random.shuffle(imageModifiersArtist)
    random.shuffle(imageModifiersMedium)
    prompt = prompt + " in the style of " + imageModifiersArtist[0] + " as a " + imageModifiersMedium[0]

    prompt = f"{prompt} for the following concept: {phrase}"

    logger.info("Generating image...")
    logger.info("image prompt: " + prompt)

    # use openai to generate a picture based on the summary
    try:
        responseImage = client.images.generate(
            prompt= prompt,
            n=4,
            size="512x512")
    except Exception as e:
        print("\n\n\n")
        print(e)
        print("\n\n\n")
        raise (e)
        
    loggerTrace.debug("responseImage: " + str(responseImage))

    image_url = [responseImage.data[0].url] * 4
    image_url[1] = responseImage.data[1].url
    image_url[2] = responseImage.data[2].url
    image_url[3] = responseImage.data[3].url

    return image_url, imageModifiersArtist[0], imageModifiersMedium[0]

# ----------------------
# reformat image(s) for display
#    return the new file name
#
def postProcessImages(imageURLs, imageArtist, imageMedium, keywords, timestr):
    # save the images from a urls into imgObjects[]
    imgObjects = []
    for numURL in range(len(imageURLs)):

        fileName = "history/" + "image" + str(numURL) + ".png"
        urllib.request.urlretrieve(imageURLs[numURL], fileName)

        img = Image.open(fileName)

        imgObjects.append(img)

    # combine the images into one image
    #widths, heights = zip(*(i.size for i in imgObjects))
    total_width = 512*2
    max_height = 512*2 + 50
    new_im = Image.new('RGB', (total_width, max_height))
    locations = [(0,0), (512,0), (0,512), (512,512)]
    count = -1
    for loc in locations:
        count += 1
        new_im.paste(imgObjects[count], loc)

    # add text at the bottom
    imageCaption = keywords + " as a " + imageMedium + " by " + imageArtist
    draw = ImageDraw.Draw(new_im)
    draw.rectangle(((0, new_im.height - 50), (new_im.width, new_im.height)), fill="black")
    font = ImageFont.truetype("arial.ttf", 18)
    # decide if text will exceed the width of the image
    #textWidth, textHeight = font.getsize(text)
    draw.text((10, new_im.height - 30), imageCaption, (255,255,255), font=font)

    # save the combined image
    newFileName = "history/" + timestr + "-image" + ".png"
    new_im.save(newFileName)

    return newFileName

# ----------------------
# generate error message image for display
#    return the new file name
#
def generateErrorImage(e, timestr):
    # make an image to display the error
    total_width = 512*2
    max_height = 512*2 + 50
    new_im = Image.new('RGB', (total_width, max_height))
    draw = ImageDraw.Draw(new_im)
    draw.rectangle(((0, 0), (new_im.width, new_im.height)), fill="black")
    
    # add error text
    imageCaption = str(e)
    
    font = ImageFont.truetype("arial.ttf", 24)
    # decide if text will exceed the width of the image
    #textWidth, textHeight = font.getsize(text)

    import textwrap
    lines = textwrap.wrap(imageCaption, width=60)  #width is characters
    y_text = new_im.height/2
    for line in lines:
        #width, height = font.getsize(line)
        #draw.text(((new_im.width - width) / 2, y_text), line, font=font) 
        #y_text += height
        height = 25
        draw.text((100, y_text), line, font=font) 
        y_text += height

    #draw.text((10, new_im.height/2), imageCaption, (255,255,255), font=font)

    # save the new image
    newFileName = "history/" + timestr + "-image" + ".png"
    new_im.save(newFileName)

    return newFileName

'''
#early experimental code follows

import tkinter as tk
from PIL import ImageTk, Image
import os

# Global reference to the window
g_windowForImage = None

def create_window(image_path):
    global g_windowForImage
    g_windowForImage = tk.Toplevel(root)
    g_windowForImage.geometry("+500+500")  # Position at (500, 500)

    # Open an image file
    img = Image.open(image_path)
    # Convert the image to a PhotoImage
    img = ImageTk.PhotoImage(img)
    # Create a label and add the image to it
    label = tk.Label(g_windowForImage, image=img)
    label.image = img  # Keep a reference to the image to prevent it from being garbage collected
    label.pack()

def close_window():
    global g_windowForImage
    if g_windowForImage is not None:
        g_windowForImage.destroy()
        g_windowForImage = None

# create image display window
root = tk.Tk()
#root.withdraw()  # Hide the root window



#end of early experimental code
'''




# ----------------------
# main program starts here
#
#
#

class processStep(IntEnum):
    NoneSpecified = 0
    Audio = 1
    Transcribe = 2
    Summarize = 3
    Keywords = 4
    Image = 5

# set up logging
logging.basicConfig(level=logging.WARNING, format=' %(asctime)s - %(levelname)s - %(message)s')

# set the OpenAI API key
#raise Exception("The 'openai.api_key_path' option isn't read in the client API. 
# You will need to pass it when you instantiate the client, 
# e.g. 'OpenAI(api_key_path='creepy photo secret key')'")

'''Traceback (most recent call last):
  File "/Users/jschrempp/Documents/devschrempp/GitHub/jschrempp.speech2picture/pyspeech.py", line 97, in <module>
    client = OpenAI(api_key_path='creepy photo secret key')
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: OpenAI.__init__() got an unexpected keyword argument 'api_key_path'
'''

# check for running over ssl to a remote machine
isOverRemoteSSL = False
"""
# this doesn't work. Running on RPi terminal it still says it is running over SSL
if ssl.OPENSSL_VERSION and platform.system() != "Darwin":
    isOverRemoteSSL = True
    print("Running over SSL to a remote machine")
else:
    print("Not running over SSL to a remote machine")
"""

# create a directory if one does not exist
if not os.path.exists("history"):
    os.makedirs("history")




# parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--savefiles", help="save the files", action="store_true") # optional argument
parser.add_argument("-d", "--debug", help="0:info, 1:prompts, 2:responses", type=int) # optional argument
parser.add_argument("-w", "--wav", help="use audio from file", type=str, default=0) # optional argument
parser.add_argument("-t", "--transcript", help="use transcript from file", type=str, default=0) # optional argument
parser.add_argument("-T", "--summary", help="use summary from file", type=str, default=0) # optional argument
parser.add_argument("-k", "--keywords", help="use keywords from file", type=str, default=0) # optional argument
parser.add_argument("-i", "--image", help="use image from file", type=str, default=0) # optional argument
parser.add_argument("-o", "--onlykeywords", help="use audio directly without extracting keywords", action="store_true") # optional argument
args = parser.parse_args()

# set the debug level
logger.setLevel(logging.INFO)

if args.debug == 1:
    logger.setLevel(logging.DEBUG)
    logger.debug("Debug level set to show prompts")
elif args.debug == 2:
    logger.setLevel(logging.DEBUG)
    loggerTrace.setLevel(logging.DEBUG)
    logger.debug("Debug level set to show prompts and response JSON")


# if we're given a file via the command line then start at that step
# check in reverse order so that processStartStep will be the latest step for any set of arguments
firstProcessStep = processStep.NoneSpecified
if args.image != 0: 
    firstProcessStep = processStep.Image
elif args.keywords != 0: 
    firstProcessStep = processStep.Keywords
elif args.summary != 0: 
    firstProcessStep = processStep.Summarize
elif args.transcript != 0: 
    firstProcessStep  = processStep.Transcribe
elif args.wav != 0:
    firstProcessStep = processStep.Audio

# if set, then record only 10 seconds of audio and use that for the keywords
g_isAudioKeywords = False
if args.onlykeywords:
    g_isAudioKeywords = True

# if true, don't ask user for input, rely on hardware buttons
g_isUsingHardwareButtons = False

# if true, we had an error and want to just go back to the top of the loop
g_abortThisIteration = False

# ----------------------
# Main Loop 
#

done = False  # set to true to exit the loop
loopDelay = 60 # delay between loops in seconds

while not done:

    if firstProcessStep > processStep.Audio or args.wav != 0:

        # we have file parameters, so only loop once
        numLoops = 1
        loopDelay = 1   # no delay if we're not looping XXX

    else:
        # no command line input parameters so prompt the user for a command

        if not g_isUsingHardwareButtons: 
            # print menu
            print("\r\n\n\n")
            print("Commands:")
            print("   o: Once, record and display; default")
            print("   a: Auto mode, record, display, and loop")
            if not g_isMacOS:
                # running on RPi
                print("   h: Hardware control")
            print("   q: Quit")

            # wait for the user to press a key
            inputCommand = input("Type a command ...")

        if inputCommand == 'h':
            # not in the menu except on RPi
            # don't ask the user for input again, rely on hardware buttons
            g_isUsingHardwareButtons = True
            print("\r\nHardware control enabled")

        elif inputCommand == 'q': # quit
            done = True
            numLoops = 0
            loopDelay = 0

        elif inputCommand == 'a': # auto mode
            numLoops = loopsMax
            print("Will loop: " + str(numLoops) + " times")
            
        else: # default is once
            numLoops = 1
            loopDelay = 0
            firstProcessStep = processStep.Audio

        if g_isUsingHardwareButtons:
            isButtonPressed = False
            while not isButtonPressed:
                # running on RPi
                # read gpio pin, if pressed, then do a cycle of keyword input
                if GPIO.input(g_goButton) == g_buttonPressed:
                    g_isAudioKeywords = True
                    numLoops = 1
                    isButtonPressed = True

        if g_isAudioKeywords:
            # we are not going to extract keywords from the transcript
            duration = 10

    # loop will normally process audio and display the images
    # but if we're given a file then start at that step (processStep)
    # and numLoops should be 1
    for i in range(0, numLoops, 1):

        g_abortThisIteration = False

        # format a time string to use as a file name
        timestr = time.strftime("%Y%m%d-%H%M%S")

        soundFileName = ""
        transcript = ""
        summary = ""
        keywords = ""
        imageURLs = ""

        # Audio - get a recording.wav file
        if firstProcessStep <= processStep.Audio:

            changeBlinkRate(constBlinkAudioCapture)

            if args.wav == 0:
                # record audio from the default microphone
                soundFileName = recordAudioFromMicrophone()

                if args.savefiles:
                    #copy the file to a new name with the time stamp
                    shutil.copy(soundFileName, "history/" + timestr + "-recording" + ".wav")
                    soundFileName = "history/" + timestr + "-recording" + ".wav"
        
            else:
                # use the file specified by the wav argument
                soundFileName = args.wav
                logger.info("Using audio file: " + args.wav)

            changeBlinkRate(constBlinkStop)
    
        # Transcribe - set transcript
        if firstProcessStep <= processStep.Transcribe:
        
            changeBlinkRate(constBlink1)

            if args.transcript == 0:
                # transcribe the recording
                transcript = getTranscript(soundFileName)

                if args.savefiles:
                    f = open("history/" + timestr + "-rawtranscript" + ".txt", "w")
                    f.write(transcript)
                    f.close()
            else:
                # use the text file specified 
                transcriptFile = open(args.transcript, "r")
                # read the transcript file
                transcript = transcriptFile.read()
                logger.info("Using transcript file: " + args.transcript)

            changeBlinkRate(constBlinkStop)

        # Summary - set summary
        if firstProcessStep <= processStep.Summarize:

            """ Skip summarization for now
            changeBlinkRate(constBlink2)

            if args.summary == 0:
                # summarize the transcript
                summary = getSummary(transcript)

                if args.savefiles:
                    f = open("history/" + timestr + "-summary" + ".txt", "w")
                    f.write(summary)
                    f.close()

            else:
                # use the text file specified by the transcript argument
                summaryFile = open(summaryArg, "r")
                # read the summary file
                summary = summaryFile.read()
                logger.info("Using summary file: " + summaryArg)
            
            changeBlinkRate(constBlinkStop)
            """


        # Keywords - set keywords
        if firstProcessStep <= processStep.Keywords:

            changeBlinkRate(constBlink3)

            if not g_isAudioKeywords:

                if args.keywords == 0:
                    # extract the keywords from the summary
                    keywords = getAbstractForImageGen(transcript) 

                    if args.savefiles:
                        f = open("history/" + timestr + "-keywords" + ".txt", "w")
                        f.write(keywords)
                        f.close()

                else:
                    # use the extract file specified by the extract argument
                    summaryFile = open(args.keywords, "r")
                    # read the summary file
                    keywords = summaryFile.read()
                    logger.info("Using abstract file: " + args.keywords)
                
            else:
                # use the transcript as the keywords
                keywords = transcript

            changeBlinkRate(constBlinkStop)

        # Image - set imageURL
        if firstProcessStep <= processStep.Image:

            changeBlinkRate(constBlink4)

            if args.image == 0:

                # use the keywords to generate images
                try:
                    imagesInfo = getImageURL(keywords)

                    imageURLs = imagesInfo[0]
                    imageArtist = imagesInfo[1]
                    imageMedium = imagesInfo[2]   

                    newFileName = postProcessImages(imageURLs, imageArtist, imageMedium, keywords, timestr)

                    imageURLs = "file://" + os.getcwd() + "/" + newFileName
                    logger.debug("imageURL: " + imageURLs)

                except Exception as e:

                    print ("AI Image Error: " + str(e))

                    newFileName = generateErrorImage(e, timestr)

                    imageURLs = "file://" + os.getcwd() + "/" + newFileName
                    logger.debug("Error Image Created: " + imageURLs)       
            
            else:
                imageURLs = [args.image]
                newFileName = args.image
                logger.info("Using image file: " + args.image )

            changeBlinkRate(constBlinkStop)
            
        # Display - display imageURL
        
        if isOverRemoteSSL:
            # don't try to disply
            print("Not displaying image because we're running over SSL")
        else:
            # display the image
            changeBlinkRate(constBlinkSlow)
            logger.info("Displaying image...")

            # display the image with pillow
            image = Image.open(newFileName)
            image.show()

            ''' Experimenting with control of the image display window
            # When it's time to display the image:
            create_window(newFileName)

            # delay 10 seconds 
            time.sleep(10)

            # When it's time to close the window:
            close_window()
            '''
            
            changeBlinkRate(constBlinkStop)

        # The end of the for loop
        changeBlinkRate(constBlinkStop)
        # are we running the command line file args?
        if firstProcessStep > processStep.Audio or args.wav != 0:
            # We've done one and we're all done
            done = True
        else:
            #delay before the next for loop iteration
            if not g_isUsingHardwareButtons:
                print("delaying " + str(loopDelay) + " seconds...")
                time.sleep(loopDelay)
            
        # end of loop

# all done
if not g_isMacOS:
    # running on RPi
    # Stop the LED thread
    changeBlinkRate(constBlinkDie)
    led_thread1.join()

    # Clean up the GPIO pins
    GPIO.cleanup()

# exit the program
print("\r\n")
exit()





