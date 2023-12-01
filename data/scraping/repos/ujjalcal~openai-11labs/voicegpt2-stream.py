import gradio as gr
import openai
from elevenlabs import generate, stream, set_api_key
from pydub import AudioSegment
from pydub.playback import play
import time
import asyncio

from config import OPENAI_API_KEY, ELEVENLABS_API_KEY

# Set API keys
openai.api_key = OPENAI_API_KEY
set_api_key(ELEVENLABS_API_KEY)

#messages = ["You are a  mortgage ai assistant. Your name is Grace. Please respond to all input in 25 words or less."]
messages = ["You are a  mortgage ai assistant named Grace. Please respond in a manner feels like the customer is talking to his or her best friend and in 25 words or less."]

## secondary function to chunk text ##
def text_chunks(input_text, chunk_size=25):  
    words = input_text.split()
    for i in range(0, len(words), chunk_size):
        yield ' '.join(words[i:i+chunk_size])

## secondary function to stream audio ##
async def gpt3_streamed_response(input_text):
    for chunk in text_chunks(input_text):
        start_time = time.time()
        print(f"Sending to GPT-3: {chunk}")
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=chunk,
            max_tokens=80,
            n=1,
            stop=None,
            temperature=0.5
        )
        elapsed_time = time.time() - start_time
        print(f"Time taken for openai.Completion.create(): {elapsed_time:.2f} seconds.")
        yield response["choices"][0]["text"]


## primary function to transcribe voice and get response from gpt ##
async def transcribe(audio):
    global messages

    audio_file = open(audio, "rb")
    start_time = time.time()
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    elapsed_time = time.time() - start_time
    print(f"Time taken for openai.Audio.transcribe(): {elapsed_time:.2f} seconds.")
    messages.append(f"\nUser: {transcript['text']}")

    # Get GPT-3 responses in a streaming manner.
    async for response_text in gpt3_streamed_response(transcript['text']):
        audio_data = generate(
            text=response_text,
            voice="Rose",
            model="eleven_monolingual_v1",
            stream=True
        )
        stream(audio_data)  # Assuming this will play the audio
        messages.append(response_text)

    chat_transcript = "\n".join(messages)
    return chat_transcript


##  UI interface ##
iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(source="microphone", type="filepath", placeholder="Please start speaking...", duration=10),  
    outputs="text",
    title="🤖 My Desktop ChatGPT Assistant 🤖",
    description="🌟 Please ask me your question and I will respond both verbally and in text to you...",
    live=True
)

iface.launch()
