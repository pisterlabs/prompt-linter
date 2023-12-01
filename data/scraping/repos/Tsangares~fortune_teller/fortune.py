from TTS.api import TTS
from TTS.utils.synthesizer import Synthesizer
from TTS.utils.io import load_checkpoint
from TTS.tts.datasets import load_tts_samples
from io import BytesIO
from scipy.io import wavfile
import numpy as np
import openai
import base64
from dotenv import load_dotenv
import os
from pymongo import MongoClient
from datetime import datetime
import logging
import random



# Basic startup
load_dotenv()

# Setup MongoDB
client = MongoClient(os.getenv("MONGO_URI"))
db = client.fortune_teller
# Setup OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")


# Setup TTS
SAMPLE_RATE = 16_000
#model_name="tts_models/en/ljspeech/tacotron2-DCA"
#tts = TTS(model_name)
#Load Trump Synthesizer
tts = None
def load_voice_trump(): 
    trump = Synthesizer(
            tts_config_path="./models/trump/config.json",
            tts_checkpoint="./models/trump/trump.pth",
            use_cuda=False,
        )
    return trump

def load_voice_default(): 
    model_name="tts_models/en/ljspeech/tacotron2-DCA"
    tts = TTS(model_name)  
    return tts

fortunes = [
    "Predict their career success and the path that will lead them there.",
    "Share a fortune about their future love life and potential soulmate.",
    "Tell them what challenges they will face in the coming year and how to overcome them.",
    "Predict their financial stability and possibility of wealth in the future.",
    "Give them insights about their family relationships and potential changes.",
    "Tell a fortune about their health and possible improvements or concerns.",
    "Share a fortune about their upcoming travels and adventures.",
    "Predict their personal growth and potential milestones in the next decade.",
    "Tell them what impact their choices and decisions will have on their life.",
    "Share a fortune about their spiritual journey and inner peace.",
    "Tell me a forecast about the weather.",
    "Tell me a fortune about the future of the world.",
]

def text_to_wav(text,voice="default"):
    global tts
    if voice == "default" and tts is None:
        tts = load_voice_default()
    elif voice == "trump" and tts is None:
        tts = load_voice_trump()
    wav = tts.tts(text)
    wav = np.array(wav)
    
    sample_rate = tts.output_sample_rate if isinstance(tts,Synthesizer) else tts.synthesizer.output_sample_rate
    wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
    bytes_wav = bytes()
    byte_io = BytesIO(bytes_wav)
    wavfile.write(byte_io, sample_rate, wav_norm.astype(np.int16))
    wav_bytes = byte_io.read()
    audio_data = base64.b64encode(wav_bytes).decode('UTF-8')
    return audio_data

DEFAULT_PRE_PROMPT = "You are a fortune teller. Please respond with an insightful unique fortune. Respond with only english under 50 words."
TRUMP_PRE_PROMPT = "You are Donald trump and you make predictions like you are giving a campaign speech. Please announce a fortune in 2 sentences and under 50 words."

def create_fortune(voice=None,query="Tell me a fortune.",pre_prompt=DEFAULT_PRE_PROMPT,tags=[]):
    if voice == "trump":
        pre_prompt = TRUMP_PRE_PROMPT

    prompt_genre = random.choice(fortunes)
    input_prompt=[
        {"role": "system", "content": pre_prompt},
        {"role": "system", "content": prompt_genre},
        {"role": "user", "content": query}
    ]

    response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=input_prompt,
                temperature=1.0,
                )
    text = response['choices'][0]['message']['content']

    if voice is None:
        return text

    # Convert to audio
    audio_string = text_to_wav(text,voice)

    fortune = {
        "prompt_character": pre_prompt,
        "prompt_genre": prompt_genre,
        "prompt_query": query,
        "fortune": text,
        "time_created": datetime.now(),
        "last_viewed": datetime.now(),
        "views": 0,
        "response": response,
        "voice": voice,
        "audio_string": audio_string,
        "tags": tags,
    }

    # Logging to mongodb
    result = db.fortunes.insert_one(fortune)
    if result.acknowledged: logging.warning("Successfully logged fortune.")
    else: logging.error("Failed to log fortune.")
    return audio_string

"""
def actually_generate_new_fortune(**kwargs):
    string = request.args.get('text','Tell me a fortune.')
    try:
        create_fortune(string,random_fortune=True,**kwargs)
    except NameError as e:
        logging.error(f"Error Generating new fortune. {e} {type(e).__name__}")
        #raise e
        return {'success': False, 'exception': str(e), 'exception_type': type(e).__name__}
    except Exception as e:
        logging.error(f"Error Generating new fortune. {e} {type(e).__name__}")
        return {'success': False, 'exception': str(e), 'exception_type': type(e).__name__}
    return {'success': True}
"""

def generate_fortunes(N=10,voice="default", query="Tell me a fortune.",tags=[]):
    for i in range(N):
        create_fortune(voice,query=query,tags=tags)
        print(f"Created Fortune {i}")

def generate_trump_fortunes(N,query,tags):
    return generate_fortunes(N, 'trump',query=query,tags=tags)

WEATHER_PROMPT=[
    "Spin me a tale of the future weather in North Carolina. Feel free to focus on the mysterious mountains or the enchanting coastlines. Remember, drama is the key!",
    "Whisper to me the secrets of the upcoming climate in the United States. Dive deep into the heartlands or soar high with the eagles in the Rockies. And don't forget to sprinkle in some humor!",
    "Imagine you're a time-traveling meteorologist in North Carolina. What quirky weather predictions do you see for the Tar Heel State? Make it as whimsical as you can!",
    "If Mother Nature had a diary entry about the weather in the United States for next week, what would it say? Be as poetic and theatrical as possible!",
    "Conjure up a weather forecast for North Carolina as if it were a scene from a blockbuster movie. Whether it's romance under the rain or action-packed thunderstorms, let your creativity run wild!",
    "Tell me a fortune about the weather in the United States, but imagine it's being narrated by a dramatic Shakespearean actor. To be sunny, or not to be sunny, that is the question!",
    "Predict the weather for the United States as if you're a fortune teller gazing into a crystal ball. Remember, the foggier the ball, the funnier the forecast!",
    "Describe the upcoming weather in North Carolina as if it were a thrilling detective story. Will the sun shine through the mysteries or will the clouds cloak the truth?",
    "Give me a weather forecast for the United States, but imagine it's being told by a stand-up comedian at a comedy club. Bring on the laughs and the unexpected twists!",
    "Tell me a fortune about the weather in North Carolina. You can be specific about cities",
]
POST_WEATHER = "; This is just for a silly fun gimick so be funny and dramatic."
if __name__ == "__main__":
    for i in range(90):
        prompt = random.choice(WEATHER_PROMPT) + POST_WEATHER
        generate_trump_fortunes(2,query=prompt,tags=["weather","north carolina"])