import os
import openai
from dotenv import load_dotenv
from nr_openai_observability import monitor

load_dotenv()

monitor.initialization()

# Set up the OpenAI API client
openai.api_key = os.getenv("OPENAI_API_KEY")

file = open("/home/varun/Downloads/david.mp3", "rb")
transcription = openai.Audio.transcribe("whisper-1", file)
print(transcription)


# import os

# import openai
# from nr_openai_observability import monitor
# from dotenv import load_dotenv
# load_dotenv()
# os.environ["NEW_RELIC_LICENSE_KEY"] = "40efa7bf334917db4467aba7398ff2d7267aNRAL" 

# monitor.initialization()
# openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.Completion.create(
#     model="text-davinci-003",
#     prompt="What is Observability?",
#     max_tokens=20,
#     temperature=0 
# )
