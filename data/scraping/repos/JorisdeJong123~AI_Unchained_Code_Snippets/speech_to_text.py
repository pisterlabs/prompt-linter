import openai
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain import LLMChain
from prompts import prompt

# Load env files
load_dotenv()
openai_api_key = os.environ.get('OPENAI_API_KEY')

# Transcribe audio with Whisper API
audio_file_path = "path/to/your/audio/file"
transcript_raw = openai.Audio.transcribe("whisper-1", file=audio_file_path)

# Create LLM
llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0.3)

# Create prompt
prompt_with_transcript = prompt.format(transcript=str(transcript_raw))

# Create chain
chain = LLMChain(llm=llm, prompt=prompt_with_transcript)

# Run chain
summary = chain.run()
