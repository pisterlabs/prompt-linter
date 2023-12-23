import os
from dotenv import load_dotenv
import cv2
from openai import OpenAI
import time
import base64
import requests



# Load the .env file
load_dotenv()

# Retrieve the API_KEY
api_key = os.getenv("API_KEY")

client = OpenAI(
        # Get the API key from the environment variable
    api_key = os.getenv('API_KEY')
)

def encode_frame_to_base64(frame):
    try:
        # Ensure frame is encoded to JPEG format before converting to base64
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            raise ValueError("Could not encode the frame into a buffer.")
        base64_image = base64.b64encode(buffer).decode('utf-8')
        return base64_image
    except Exception as e:
        print(f"Error encoding frame: {e}")
        return None

video_file_path = 'curl.mp4'
video_capture = cv2.VideoCapture(video_file_path)

frame_count = 0

# Function to send the image to the OpenAI API and get the description
def get_image_description(base64_image):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Format the base64 string as a data URI
    data_uri = f"data:image/jpeg;base64,{base64_image}"

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "You are using this image and the following to analyze the form."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_uri
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    description_text(response.json())

def description_text(x):    # Extracting the 'content' from the first 'choice' in the 'choices' list,
    # no matter what the 'id' is
    if 'choices' in x and x['choices']:
        # This checks if 'choices' exists and is not empty
        first_choice = x['choices'][0]
        
        if 'message' in first_choice and 'content' in first_choice['message']:
            # Assuming each 'choice' has a 'message' with 'content'
            content_text = first_choice['message']['content']
            print(content_text)
            speech(content_text)
        else:
            print("The 'message' or 'content' key is missing in the first choice.")
    else:
        print("The 'choices' key is missing in the response or is empty.")

def speech(content_text):
    # Output file path for the MP3
    speech_file_path = "speech.mp3"

    # Call the OpenAI API to generate speech
    response = client.audio.speech.create(
        model="tts-1",
        voice="echo",
        input=content_text
    )

    # Save the generated speech to an MP3 file
    with open(speech_file_path, 'wb') as f:
        f.write(response.content)


while video_capture.isOpened():
    ret, frame = video_capture.read()

    if not ret:
        print("Could not read frame or end of video reached.")
        break

    if frame_count % 10 == 0:
        base64_image = encode_frame_to_base64(frame)
        if base64_image is not None:
            get_image_description(base64_image)

    frame_count += 1

video_capture.release()

