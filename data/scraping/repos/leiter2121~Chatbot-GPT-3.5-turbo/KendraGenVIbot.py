import openai
import boto3
import tempfile
import pygame
import requests
import os
import pickle  # Added pickle module for serialization
from pygame import mixer

# Set your OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"
subscription_key = 'YOUR_BING_SEARCH_API_KEY'
endpoint = 'BING SEARCH ENDPOINT'

# Set your Amazon Polly credentials
polly = boto3.client(
    "polly",
    region_name="YOUR_REGION_NAME",
    aws_access_key_id="YOUR_AWS_ID_API_KEY",
    aws_secret_access_key="YOUR_AWS_SECRET_KEY"
)

# Check if conversation history pickle file exists, if not, create an empty list
if not os.path.exists("conversation_history.pickle"):
    with open("conversation_history.pickle", "wb") as f:
        pickle.dump([], f)

# Load conversation history from pickle file
with open("conversation_history.pickle", "rb") as f:
    conversation = pickle.load(f)

def search_bing(query):
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key
    }
    params = {
        'q': query,
        'count': 5,  # Number of search results to retrieve
        'offset': 0,  # Offset for pagination
        'mkt': 'en-US'  # Market code for search results (e.g., 'en-US' for English)
    }
    response = requests.get(endpoint, headers=headers, params=params)
    result = response.json()
    
    # Extract search results as a string
    search_results = ""
    for item in result.get('webPages', {}).get('value', []):
        search_results += f"{item['name']}: {item['url']}\n"
    
    return search_results

def generate_chat_response(conversation):
    # Truncate or omit conversation history to fit within 4096 tokens
    conversation = conversation[-15:]  # Limit conversation history to last 50 messages
    conversation_tokens = sum(len(message['content'].split()) for message in conversation)
    if conversation_tokens > 4096:
        conversation = conversation[-1:]  # Only keep the last message if it exceeds 4096 tokens

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation,
        max_tokens=1000  # Update this value to limit the total token count of conversation
    )
    chat_reply = response['choices'][0]['message']['content']
    return chat_reply.strip()

def generate_dalle_image(prompt, size="1024x1024"):
    response = openai.Image.create(
        prompt=prompt,
        size=size
    )
    image_url = response['data'][0]['url']
    return image_url

def speak_text(text):
    response = polly.synthesize_speech(
        Text=text,
        OutputFormat="mp3",
        VoiceId="Joey",
        Engine="neural"
    )
    
    with tempfile.TemporaryFile() as temp_file:
        temp_file.write(response["AudioStream"].read())
        temp_file.flush()
        temp_file.seek(0)
        mixer.init()
        mixer.music.load(temp_file)
        mixer.music.play()
        while mixer.music.get_busy():
            continue
        
while True:
    user_input = input("You: ")
    conversation.append({"role": "user", "content": user_input})
    if user_input.lower() == "quit":
        speak_text("Goodbye!")
        # Save conversation history to pickle file before quitting
        with open("conversation_history.pickle", "wb") as f:
            pickle.dump(conversation, f)
        break
    if "search" in user_input.lower():
        # Ask the user to search for something
        prompt = "What would you like to search for:"
        speak_text(prompt)
        conversation.append({"role": "assistant", "content": prompt})
        return_value = search_bing(user_input)
        speak_text(return_value)
        print(f"Kendra: {return_value}")
        conversation.append({"role": "system", "content":"You just searched Bing for " + user_input + " and got the following results"})
        conversation.append({"role": "user", "content": user_input})
        conversation.append({"role": "assistant", "content": return_value})
        continue
    
    if "generate image" in user_input.lower():
        # Ask the user to describe the image they want to see
        prompt = "Please describe the image you want to see:"
        speak_text(prompt)
        print("Kendra: " + prompt)
        conversation.append({"role": "system", "content": "You are generating an image using DALL·E"})
        conversation.append({"role": "user", "content": user_input})
        conversation.append({"role": "assistant", "content": prompt})
    elif len(conversation) > 1 and conversation[-2]["role"] == "assistant" and "describe the image" in conversation[-2]["content"]:
        # Use the user's description as the prompt to generate an image with DALL·E
        image_prompt = user_input
        image_url = generate_dalle_image(image_prompt)

        # Download the image from URL
        response = requests.get(image_url)
        file_name = user_input.replace(" ", "_") + ".jpg"
        with open(os.path.join(os.path.expanduser("~/Desktop"), file_name), "wb") as f:
            f.write(response.content)
            speak_text("The generated image has been saved to your desktop!")
            print(f"Kendra: The generated image has been saved to your desktop as '{file_name}'")
    else:
        # Generate a text response with chat model
        chat_reply = generate_chat_response(conversation)
        print("Kendra: " + chat_reply)
        conversation.append({"role": "assistant", "content": chat_reply})

        # Use Amazon Polly to convert bot's reply to speech
        speak_text(chat_reply)
