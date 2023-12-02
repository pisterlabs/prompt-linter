import openai, os # TODO: dotenv.load_dotenv(); apikey = os.getenv('apikey' || apikey_default); initialize api key

# T


MODEL_NAME = 'whisper-1'

def transcribe_audio(directory=None):
    for file in os.listdir(directory):
        f = os.path.join(directory, file)
        if os.path.isfile(f):
             audio_file = open(f, 'rb')
             transcript = openai.Audio.transcribe(MODEL_NAME, audio_file)
             return transcript

