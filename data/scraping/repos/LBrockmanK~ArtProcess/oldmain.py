import os
import shutil
import json
import openai
from PIL import Image
import piexif
import base64
import requests
import math

# Function to encode the image and get tags from OpenAI API
def classify_image(image_path, api_key):
    # Encode the image to base64
    base64_image = encode_image(image_path)

    # Headers for the API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Payload for the API request
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What’s in this image?"},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
                ],
            }
        ],
        "max_tokens": 300
    }

    # Make the API request
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    # Handle the response
    try:
        response_data = response.json()
        print("API Response:", response_data)  # Debug: Print the full response
        # Parse the response to get the tags
        tags_text = response_data['choices'][0]['message']['content']['text']
        tags = tags_text.split(', ')  # Split tags by comma and space
        return tags
    except Exception as e:
        print(f"Error processing the response: {e}")
        return []

# Helper function to encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# Function to embed tags into image metadata
def embed_metadata(image_path, tags):
    image = Image.open(image_path)
    exif_dict = piexif.load(image.info.get('exif', b''))
    
    # Convert tags list to JSON string for embedding
    user_comment = json.dumps(tags)
    exif_dict['Exif'][piexif.ExifIFD.UserComment] = piexif.helper.UserComment.dump(user_comment)
    
    exif_bytes = piexif.dump(exif_dict)
    image.save(image_path, exif=exif_bytes)

# Function to move image to the Outputs folder
def move_to_output(image_path):
    output_folder = os.path.join('Outputs')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    shutil.move(image_path, os.path.join(output_folder, os.path.basename(image_path)))

# Function to resize the image if it exceeds a certain number of pixels
def resize_image_if_needed(image_path, max_pixels=65536):
    image = Image.open(image_path)
    width, height = image.size
    current_pixels = width * height

    if current_pixels > max_pixels:
        scale_factor = math.sqrt(max_pixels / current_pixels)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)
        resized_image.save(image_path)

# Helper function to encode image to base64
def encode_image(image_path):
    resize_image_if_needed(image_path)  # Resize the image before encoding
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Main processing loop for images in the Inputs folder
def process_images(input_folder, api_key):
    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        if os.path.isfile(image_path):
            try:
                tags = classify_image(image_path, api_key)  # Pass the api_key argument here
                embed_metadata(image_path, tags)
                move_to_output(image_path)
                print(f"Processed and moved: {image_name}")
            except Exception as e:
                print(f"Error processing {image_name}: {e}")

# Define the root, input, and output folders
root_folder = os.getcwd()
input_folder = os.path.join(root_folder, 'Inputs')

# Retrieve API key from environment variable
api_key = os.getenv('OPENAI_API_KEY')  # Make sure the environment variable is set

# Check if the API key is available
if api_key:
    # Process all images in the Inputs folder
    process_images(input_folder, api_key)
else:
    print("API key not found. Please set the OPENAI_API_KEY environment variable.")