import requests
import shutil
import json
import os
import openai
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Replace with your actual APP_KEY, APP_SECRET and OPENAI_KEY
APP_KEY = os.environ["APP_KEY"]
APP_SECRET = os.environ["APP_SECRET"]
openai.api_key = os.environ["OPENAI_KEY"]
input_filename = "Raw Audio.mp3"


# Retrieve access token for authentication
payload = {"grant_type": "client_credentials", "expires_in": 1800}
response = requests.post(
    "https://api.dolby.io/v1/auth/token",
    data=payload,
    auth=requests.auth.HTTPBasicAuth(APP_KEY, APP_SECRET),
)
body = json.loads(response.content)
access_token = body["access_token"]

# Use the access_token for further API requests
print(">> Access token Generated")
print("")

# Specify the file path of the input media file
file_path = os.path.join(os.path.dirname(__file__), "input", input_filename)

# Declare your dlb:// location
url = "https://api.dolby.com/media/input"
headers = {
    "Authorization": "Bearer {0}".format(access_token),
    "Content-Type": "application/json",
    "Accept": "application/json",
}

# Create the input media request
body = {
    "url": "dlb://Enhanced Audio.mp3",
}

# Send the input media request
response = requests.post(url, json=body, headers=headers)
response.raise_for_status()
data = response.json()
presigned_url = data["url"]
print(">> Pre-signed URL Generated")
print("")

# Upload the input media file to the pre-signed URL
print("Uploading audio file from {0} to {1}".format(file_path, "server"))
print("")

with open(file_path, "rb") as input_file:
    requests.put(presigned_url, data=input_file)
print(">> Audio File Uploaded to the server")
print("")

# Enhance the audio
body = {
    "input": "dlb://new.mp3",
    "output": "dlb://out/newaudio.mp3",
    "content": {"type": "mobile_phone"},
}

url = "https://api.dolby.com/media/enhance"

# Send the audio enhancement request
response = requests.post(url, json=body, headers=headers)
response.raise_for_status()
print("Enhancing the audio...")
print("")


# Download the enhanced audio
if not os.path.exists("output"):
    os.mkdir(os.path.join(os.path.dirname(__file__), "output"))
output_path = os.path.join(os.path.dirname(__file__), "output", "Enhanced Audio.mp3")

url = "https://api.dolby.com/media/output"

args = {
    "url": "dlb://Enhanced Audio.mp3",
}

# Send the download request for the enhanced audio
with requests.get(url, params=args, headers=headers, stream=True) as response:
    response.raise_for_status()
    response.raw.decode_content = True
    print(
        "Downloading enhanced audio from {0} into {1}".format(
            "DOLBY.IO", output_path
        )
    )
    print("")

    with open(output_path, "wb") as output_file:
        shutil.copyfileobj(response.raw, output_file)
print(">> Enhanced Audio File downloaded successfully into the output folder")
print("")

print("Transcribing the audio...")
print("")


# Transcribe the audio
audio_file = open(
    os.path.join(os.path.dirname(__file__), "output", "Enhanced Audio.mp3"), "rb"
)
transcript = openai.Audio.transcribe("whisper-1", audio_file)


# Write the output text to a file
file = open(
    os.path.join(os.path.dirname(__file__), "output", "Audio Transcription.txt"), "w"
)
file.write(transcript.text)
file.close()
print(">> Audio Transcription file downloaded successfully into the output folder")
print("")


# Perform chat completion using GPT-3.5 Turbo model
text_file = open(
    os.path.join(os.path.dirname(__file__), "output", "Audio Transcription.txt"), "rb"
)
Fcontents = text_file.read().decode("utf-8")

# List available models
models = openai.Model.list()

print(f"Summarizing the transcribed text using {models.data[0].id} model...")
print("")

# Create a chat completion
chat_completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content": "The Raw form is the original audio as spoken by the speaker. The Cleaned up form is an optimized version of the text, where ChatGPT removes unnecessary words or parts that do not contribute to the conversation. The Summary form is a shortened version of the cleaned-up text that retains the essential information. According to the instruction can you print the cleaned up and summary of this speech ",
        },
        {"role": "user", "content": Fcontents},
    ],
)

# Get the summary from the chat completion response
summary = chat_completion.choices[0].message.content


# Write the summary text to a file
file = open(os.path.join(os.path.dirname(__file__), "output", "summary.txt"), "w")
file.write(summary)
file.close()
print(">> Summary File downloaded successfully into the output folder")
print("")
