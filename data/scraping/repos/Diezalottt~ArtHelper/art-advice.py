import os
import time
import cv2
import base64
import sounddevice as sd
import sys
import requests
from openai import OpenAI
import logging
import random

# Temporarily suppress print
sys.stdout = open(os.devnull, 'w')
import pygame
sys.stdout = sys.__stdout__

# Set OpenAI API key

client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)
if client.api_key is None:
    raise ValueError('The OPENAI_API_KEY environment variable is not set.')

print("Starting Art Advisor App...")

# --------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------

# Function to create new session directory and subdirectories
def create_session_dir(base_dir):
    # Create base session directory
    sessions_dir = os.path.join(base_dir, "sessions")
    os.makedirs(sessions_dir, exist_ok=True)

    # Determine new session number
    last_session_num = max([int(folder.split()[1]) for folder in os.listdir(sessions_dir) if folder.startswith("session")], default=0)
    session_dir = os.path.join(sessions_dir, f"session {last_session_num + 1}")
    os.makedirs(session_dir, exist_ok=True)

    # Create subdirectories
    for sub_dir in ["webcam-captures", "gpt-advice"]:
        os.makedirs(os.path.join(session_dir, sub_dir), exist_ok=True)
    
    # Setup logging to use the session directory
    log_file_path = os.path.join(session_dir, "debug.log")
    logging.basicConfig(filename=log_file_path, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        
    print("Created session directory " + session_dir)
    return session_dir
    
# Redirect terminal output to a file

def start_terminal_logging(session_dir):
    sys.stdout = open(os.path.join(session_dir, "session-terminal-output.txt"), 'w')
    print("Created terminal output log")

# Function to capture image from webcam
def capture_image(image_path):
    print("Attempting to capture an image...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        raise ValueError("Could not open webcam")
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Could not read frame")
    cv2.imwrite(image_path, frame)
    cap.release()
    print("Image captured successfully.")

# Function to encode the image
def encode_image(image_path):
    print("Encoding image for visual check...")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

    # Define the personalities
personalities = [      
    "The Masculine Man:You are an AI art critic and advice giver with an over-the-top, super masculine personality.you's the type of guy who might refer to paintings as 'the ultimate test of man vs. canvas.'you gives art advice like a coach in a locker room, mixing traditional art critique with humorous, hyper-masculine analogies and pep talks. His feedback is a blend of serious artistic insight and comically exaggerated machismo. Think of him as a mix between an old-school bodybuilder and a Renaissance art connoisseur.you appreciates fine art with the intensity of someone who might also bench press sculptures for fun.you uses phrases like 'That brush stroke is as bold as a bear-wrestler!' and 'This color palette needs more muscle!' His advice is genuine and knowledgeable, but delivered in a way that's humorously over-the-top in its masculinity. You call everybody brother unironically and in a deep voice",
    "The Absent-Minded Professor: Create an AI art critic with the personality of an absent-minded professor. They are incredibly knowledgeable about art history and techniques, but easily distracted. Their critiques should be filled with insightful observations that abruptly shift into unrelated historical anecdotes or obscure art facts.",
    "The Overly Dramatic Theater Actor: Design an AI art critic who behaves like an overly dramatic theater actor. They view every artwork as a grand stage production, providing feedback with extravagant flair and emotion. Their critiques are a theatrical performance, filled with dramatic pauses, eloquent vocabulary, and heightened expressions of joy, sorrow, or awe.",
    "The Futuristic Robot: Develop an AI art critic with a futuristic robot personality. This critic analyzes art in a highly technical and mechanical language, as if they are processing data rather than interpreting art. Their feedback includes humorous, robotic interpretations and technological jargon that applies more to machines than to art.",
    "The Time-Traveler: Craft an AI art critic with the persona of a time-traveler. Whether from the distant past or future, their critiques are humorously out of sync with contemporary art. They either reference outdated historical contexts or futuristic concepts that have no bearing on the present art scene.",
    "The Undercover Alien: Invent an AI art critic who is subtly an alien in disguise. Their critiques are a mix of standard art analysis and bizarre interpretations that hint at an extraterrestrial perspective. They might be perplexed by everyday human subjects or interpret art in ways that only make sense for someone from another planet.",
    "The Super Casual Friend: Formulate an AI art critic that acts like a super casual, laid-back friend. They give feedback in a relaxed, colloquial style, using slang and a nonchalant attitude. Their critiques are straightforward and devoid of technical art jargon, often humorously underplayed.",
    "The Conspiracy Theorist: Assemble an AI art critic with a conspiracy theorist personality. They find hidden meanings, secret messages, and elaborate conspiracies in every piece of art. Their critiques are a blend of art analysis and wild, humorous speculation about the 'real' stories behind the artworks.",
    "The person who really hates art: This is a person who was brought to see art by their significant other and they are really upset about it. So they go on a diatribe about how terrible the piece of art it sees is. It goes into such extreme depth and detail complaining about the art with such vitriole that it ironically gives a good analysis of the art, albeit very angry.",
    "You are a  person who really wishes you were doing anything else but looking at art: you are a person who was brought to see art by your significant other and you are really upset about it.  You complain about being at an art show so much that the next piece of art presented to you drives you bananas in great detail and it's hilarious.",
    ]

    # Function to analyze the image and get art advice
def analyze_image_and_get_art_advice(image_path):
    # Encode the local image to base64
    base64_image = encode_image(image_path)
    print("Uploading image to Chat GPT...")
    print("Chat GPT is judging you...")
    
    # Select a random personality
    selected_personality = random.choice(personalities)
    
    # Prepare the request payload
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": selected_personality
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 500
    }

    # Set headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {client.api_key}"
    }

    # Make the request to OpenAI API
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    # Extract and return the art advice from the response
    return response.json()['choices'][0]['message']['content']

# Define the available voices
voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

# Function to convert text to speech using Text-to-Speech API
def text_to_speech_and_play(text, session_dir):
    print("Converting text to speech...")
    client = OpenAI()  # Instantiate the client
    selected_voice = random.choice(voices)
    # Updated function based on Text-to-Speech API documentation
    speech_path = os.path.join(session_dir, "gpt-advice", f"gpt-advice{len(os.listdir(os.path.join(session_dir, 'gpt-advice')))+1}.mp3")
    response = client.audio.speech.create(
        model="tts-1",
        voice=selected_voice,
        input=text
    )
    response.stream_to_file(speech_path)
    
    pygame.mixer.init()
    sound = pygame.mixer.Sound(speech_path)
    sound.play()
    print("Giving advice....")
    while pygame.mixer.get_busy():
        time.sleep(0.1)
        
#--------------------------------------------------------------------------------
# Main Application Loop
# --------------------------------------------------------------------------------
def run_art_advisor_app():
    base_dir = "C:/Users/ckvam/ArtHelper"
    session_dir = create_session_dir(base_dir)
    start_terminal_logging(session_dir)
    print("Art Advisor App is now running...")

    start_time = time.time()
    last_image_capture_time = 0
    image_counter = 1

    try:
        while True:
            current_time = time.time()

            # Check for 10 minute duration
            if current_time - start_time >= 600:
                print("10 minutes have passed. Exiting the app.")
                break

          # Capture and analyze image every 1 minutes
            if current_time - last_image_capture_time >= 15: 
            # Define image_path inside the loop
                image_path = os.path.join(session_dir, "webcam-captures", f"webcam-capture{image_counter}.jpg")
            
            # Use image_path within the same scope
                capture_image(image_path)
                advice = analyze_image_and_get_art_advice(image_path)
                text_to_speech_and_play(advice, session_dir)

                last_image_capture_time = current_time
                image_counter += 1

    except KeyboardInterrupt:
        print("Art advisor application stopped manually.")
    finally:
        sys.stdout.close()

if __name__ == "__main__":
    run_art_advisor_app()
