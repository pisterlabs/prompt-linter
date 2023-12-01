# You also need to install the MPV library https://github.com/rossy/mpv-install/blob/master/README.md if you want to enable audio collection

import azure.cognitiveservices.speech as speechsdk
import config
import elevenlabs
import openai
import time
from pathlib import Path
import importlib

filename = Path(__file__).stem
assistantname = filename.replace("_thread", "")
configname = assistantname + "_config"
threadconfig = assistantname + "_thread_conf"

threadconf = importlib.import_module(threadconfig)
assistantconf = importlib.import_module(configname)

threadid = threadconf.threadid
voiceid = assistantconf.voiceid
assistantid = assistantconf.assistantid
language = assistantconf.language

from elevenlabs import set_api_key

#Initializing Elevenlabs and OpenAI API Keys
set_api_key(config.elevenlabsapikey) # Elevenlabs API Key
openai.api_key = config.openaiapikey # OpenAI Key


# Set up Azure Speech To Text
speech_config = speechsdk.SpeechConfig(subscription=config.azureapikey, region=config.azureregion)
speech_config.speech_recognition_language=language
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)




while True:
    print("Talk now")
    result = speech_recognizer.recognize_once() #get voice input
    message = format(result.text)
    print(f"You: {message}")
    if message:
        message = openai.beta.threads.messages.create( #create message from voice input
            thread_id=threadid,
            role="user",
            content=message
        )
        run = openai.beta.threads.runs.create(
            thread_id=threadid,
            assistant_id=assistantid
        )

         #check if the thread has finished
        while True:
          running = openai.beta.threads.runs.retrieve(
            thread_id=threadid, 
            run_id=run.id
          )

          #if the thread has finished, get the list of messages
          if running.status == "completed":
            messages = openai.beta.threads.messages.list(
            thread_id=threadid
            )

            #extract the generated message from the list of all messages
            first_message = messages.data[0]   
            content=first_message.content
            value = content[0].text.value
            print(f"{assistantname}: {value}")
            audio_stream = elevenlabs.generate(text=value, voice=voiceid, model="eleven_multilingual_v2", stream=True)
            output = elevenlabs.stream(audio_stream)
            break
          time.sleep(0.5)
        
