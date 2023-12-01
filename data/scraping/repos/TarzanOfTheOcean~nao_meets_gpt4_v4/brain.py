import requests
import os
import openai
import speech_recognition as sr
from dotenv import load_dotenv
import time
import tiktoken


# setup
load_dotenv() # Load environment variables from .env file
max_response_tokens = 250
token_limit = 4096
openai.api_key = os.getenv("AZURE_OPENAI_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_type = 'azure'
openai.api_version = '2023-05-15'
BODY_URL = "http://localhost:5004"  # Assuming it runs locally


class NaoStream:
    
    def __init__(self, audio_generator):
        self.audio_generator = audio_generator

    def read(self, size=-1):  # added size parameter, default -1
        try:
            return next(self.audio_generator)
        except StopIteration:
            return b''

class NaoAudioSource(sr.AudioSource):

    def __init__(self, server_url='http://localhost:5004'): 
        self.server_url = server_url
        self.stream = None
        self.is_listening = False
        self.CHUNK = 1024
        self.SAMPLE_RATE = 16000
        self.SAMPLE_WIDTH = 2

    def __enter__(self): # this is called when using the "with" statement
        requests.post(f"{self.server_url}/start_listening")
        self.is_listening = True
        self.stream = NaoStream(self.audio_generator())  # wrap the generator
        return self # return object (self) to be used in the with statement

    def audio_generator(self): # generator function that continuously fetches audio chunks from the server as long as 'self.is_listening' is True
        while self.is_listening:
            response = requests.get(f"{self.server_url}/get_audio_chunk")
            yield response.content 
            # yield is used to return a value from a generator function, but unlike return, it doesn't terminate the function
            # instead, it suspends the function and saves its state for later resumption
            time.sleep(0.05)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.is_listening = False
        requests.post(f"{self.server_url}/stop_listening")


def get_user_text():
    
    while True:
            
        # Record audio
        filename = "input.wav"
        start = time.time()
        print("Recording...")
        with NaoAudioSource() as source:
            recognizer = sr.Recognizer()
            source.pause_threshold = 1 # seconds of non-speaking audio before a phrase is considered complete
            audio = recognizer.listen(source, phrase_time_limit=None, timeout=None)
            with open(filename, "wb") as f:
                f.write(audio.get_wav_data())
        end = time.time()
        print(f"Recording took {end - start} seconds")

        # Transcribe audio to text
        try: 
            start = time.time()
            print("Transcribing...")
            text = recognizer.recognize_google(audio)
            end = time.time()
            print(f"Transcribing took {end - start} seconds")
            break
        except sr.UnknownValueError:
            print(f"Google Speech Recognition could not understand audio, retrying...")
    
    print("You said: " + text)

    return text

def get_gpt_text(conversation_context):

    # Trim the conversation context to fit the token limit
    conversation_context = trim_context(conversation_context)

    # Process the received input with GPT-4
    start = time.time()
    response = openai.ChatCompletion.create(
        engine="NAO35",
        messages=conversation_context
    )
    end = time.time()
    print(f"{response.engine} took {end - start} seconds to respond")

    # Extract the GPT-4 response
    gpt4_message = response['choices'][0]['message']['content']

    print(f"Nao: {gpt4_message}")

    return gpt4_message

def send_gpt_text_to_body(gpt4_message):

    requests.post(f"{BODY_URL}/talk", json={"message": gpt4_message}) # Send the GPT-4 response to the body

def save_conversation(context, filename):

    with open("conversation_context.txt", "w") as f:
        for entry in conversation_context:
            role = entry['role'].capitalize()  # Capitalize the role for formatting
            content = entry['content']
            f.write(f"{role}:\n{content}\n\n")


def trim_context(context):
    """see https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/chatgpt?tabs=python-new&pivots=programming-language-chat-completions for more details."""

    def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
        """Return the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        if model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
            }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif "gpt-3.5-turbo" in model:
            print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
            return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
        elif "gpt-4" in model:
            print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
            return num_tokens_from_messages(messages, model="gpt-4-0613")
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens
    
    conv_history_tokens = num_tokens_from_messages(context)
    while conv_history_tokens + max_response_tokens >= token_limit:
        del context[1] 
        conv_history_tokens = num_tokens_from_messages(context)

    return context


import os

# Print the current working directory
print("Current Working Directory: ", os.getcwd())

# conversation loop ############################################################################################################################################################################

with open("system_prompt.txt", "r") as f:
    system_prompt = f.read() # Read system prompt from file

conversation_context = [{"role": "system", "content": system_prompt}] # Initialize conversation context with system prompt

running = True
while running:
    user_message = get_user_text() # Get the user's message

    conversation_context.append({"role": "user", "content": user_message}) # Add the user's message to the conversation context
    
    gpt4_message = get_gpt_text(conversation_context)
    
    send_gpt_text_to_body(gpt4_message)
    
    conversation_context.append({"role": "assistant", "content": gpt4_message}) # Add the GPT-4 response to the conversation context
    
    save_conversation(context=conversation_context, filename="conversation_context.txt") # Write conversation context to file for easier debugging etc.