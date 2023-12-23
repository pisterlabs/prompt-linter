##Add this line 
#from objc import super
##to the top of the file <venv_dir>/lib/python3.xxx/site-packages/pyttsx3/drivers/nsss.py
##to fix the NSSpeechDriver error in MacOSX

import os
import environment

from langchain import ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from llms import defaultLLM as llm

template = """Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

Assistant is aware that human input is being transcribed from audio and as such there may be some errors in the transcription. It will attempt to account for some words being swapped with similar-sounding words or phrases. Assistant will also keep responses concise, because human attention spans are more limited over the audio channel since it takes time to listen to a response.

{history}
Human: {human_input}
Assistant:"""

prompt = PromptTemplate(
    input_variables=["history", "human_input"], 
    template=template
)


chatgpt_chain = LLMChain(
    llm=llm, 
    prompt=prompt, 
    verbose=True, 
    memory=ConversationBufferWindowMemory(k=2),
)

import speech_recognition as sr

#For online tts
from gtts import gTTS
from playsound import playsound

#For offline tts
import pyttsx3
engine = None
# engine = pyttsx3.init()
engine = pyttsx3.init('nsss')


def listen(engine):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print('Calibrating...')
        r.adjust_for_ambient_noise(source, duration=5)
        # optional parameters to adjust microphone sensitivity
        # r.energy_threshold = 200
        # r.pause_threshold=0.5    
        
        print('Okay, go!')
        while(1):
            text = ''
            print('listening now...')
            try:
                audio = r.listen(source, timeout=5, phrase_time_limit=30)
                print('Recognizing...')
                # whisper model options are found here: https://github.com/openai/whisper#available-models-and-languages
                # other speech recognition models are also available.
                text = r.recognize_whisper(audio, model='medium.en', show_dict=True, )['text']
            except Exception as e:
                unrecognized_speech_text = f'Sorry, I didn\'t catch that. Exception was: {e}s'
                text = unrecognized_speech_text
            spokenText = "-------Recognized text is: {0}--------".format(text)
            print(spokenText)
            speak(spokenText)

            response_text = chatgpt_chain.predict(human_input=text)
            spokenText = "-------Chatgpt response text is: {0}--------".format(response_text)
            print(spokenText)
            speak(spokenText)
                

def speak(text):
    audio = gTTS(text=text, lang="en", slow=False)
    audio.save("example.mp3")
    playsound("example.mp3")

def speakTTSX3(text):
    if engine is not None:
        engine.say(text)  
        engine.runAndWait() 

# speak("What is the super string theory?")
# speakTTSX3("What is the super string theory?")
listen(engine)




# import whisper

# model = whisper.load_model("base")
# result = model.transcribe("audio.mp3")
# print(result["text"])
# import whisper

# model = whisper.load_model("base")

# # load audio and pad/trim it to fit 30 seconds
# audio = whisper.load_audio("audio.mp3")
# audio = whisper.pad_or_trim(audio)

# # make log-Mel spectrogram and move to the same device as the model
# mel = whisper.log_mel_spectrogram(audio).to(model.device)

# # detect the spoken language
# _, probs = model.detect_language(mel)
# print(f"Detected language: {max(probs, key=probs.get)}")

# # decode the audio
# options = whisper.DecodingOptions()
# result = whisper.decode(model, mel, options)

# # print the recognized text
# print(result.text)

###Commandline
## whisper audio.flac audio.mp3 audio.wav --model medium
## whisper japanese.wav --language Japanese --task translate