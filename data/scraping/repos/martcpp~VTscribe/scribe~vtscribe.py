# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai
from dotenv import load_dotenv
from scribe.noise_reduction import adjust_audio

load_dotenv()  # Load the environment variables
 
# Set your API key
openai.api_key = os.environ.get("OPENAI_API_KEY")
# Initialize the recognizer

# Now you can perform speech recognition on the adjusted audio s

audio = adjust_audio("SONd.wav")

audio_file= open(audio, "rb")
transcript = openai.Audio.transcribe("whisper-1", file=audio_file)
print(transcript)
