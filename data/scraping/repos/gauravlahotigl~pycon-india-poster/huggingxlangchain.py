# Use a pipeline as a high-level helper
from transformers import pipeline
import warnings
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import gtts
from playsound import playsound

warnings.filterwarnings('ignore')
load_dotenv()

#speech-to-text conversion
speech_to_text = pipeline("automatic-speech-recognition", model="openai/whisper-large-v2") #loading OpenAI whisper model using Huggingface
converted_text = speech_to_text('./Audio/Harohalli.m4a') #passing the recorded audio to the model
converted_text = converted_text['text'] 
print('Audio input:' + converted_text)

#getting the answer of the question using ChatGPT using Langchain
chatgpt = ChatOpenAI(temperature=0.1) #instantiating model
message = chatgpt([
    HumanMessage(content=str(converted_text))
]).content #passing the converted audio as an input ot ChatGPT
print('ChatGPT output:' + message)

#converting text to speech
# text_to_speech = pipeline("text-to-speech", model="microsoft/speecht5_tts") #loading the model
# text_to_speech(message)
tts = gtts.gTTS(message)
tts.save('tts.mp3')
playsound('tts.mp3')