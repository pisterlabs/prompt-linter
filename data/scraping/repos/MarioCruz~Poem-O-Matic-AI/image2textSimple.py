import requests
import cv2
import os
import openai
import time
import platform

# Azure setup
API_KEY = os.getenv('AZURE_API_KEY')
ENDPOINT = 'https://mario-image-to-text.cognitiveservices.azure.com/'
DESCRIBE_URL = f'{ENDPOINT}/vision/v3.1/describe'

headers = {
    'Ocp-Apim-Subscription-Key': API_KEY,
    'Content-Type': 'application/octet-stream'
}

# Initialize OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

def clear_screen():
    system_name = platform.system().lower()
    if 'windows' in system_name:
        os.system('cls')
    else:
        os.system('clear')

def capture_image():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error initializing camera. Please check if it's connected and retry.")
        return False, None

    time.sleep(2)
    ret, frame = cam.read()
    cam.release()

    if ret:
        cv2.imshow('Captured Image', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return ret, frame

def send_image_to_azure(url, image, params=None):
    _, img_encoded = cv2.imencode('.jpg', image)
    try:
        response = requests.post(url, headers=headers, data=img_encoded.tobytes(), params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
        print("Error Details:", response.text)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")
    return {}

def create_poem(description, tags):
    tag_string = ', '.join(tags[:5])
    context_info = ""
    if tags:
        context_info += f" The image has elements like {tag_string}."

    prompt = [
        {"role": "system", "content": "You are a poetic muse, able to capture the vibrant spirit and allure of Miami with your words. From the radiant sunsets over Biscayne Bay to the lively rhythms of Little Havana, Miami's essence flows through your verses."},
        {"role": "system", "content": "Remember to immerse the reader in Miami's unique atmosphere and culture."},
         {"role": "system", "content": "in the style of a sonnet"},
        {"role": "user", "content": f"Write a poem based on the image description: {description}.{context_info}"}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt,
        temperature=0.6
    )

    choices = response.get('choices', [])
    if not choices:
        print("Unexpected response from OpenAI. Couldn't generate poem.")
        return ""

    return choices[0].get('message', {}).get('content', "")

if __name__ == "__main__":
    try:
        clear_screen()

        if not API_KEY:
            print("Azure API key not found. Ensure the AZURE_API_KEY environment variable is set.")
            exit()

        if not openai.api_key:
            print("OpenAI API key not found. Ensure the OPENAI_API_KEY environment variable is set.")
            exit()

        success, captured_img = capture_image()
        if not success:
            print("Failed to grab frame from camera. Check camera availability.")
            exit()

        # Get description and tags
        response_data = send_image_to_azure(DESCRIBE_URL, captured_img)
        primary_description = response_data.get('description', {}).get('captions', [{}])[0].get('text', "")
        tags = response_data.get('description', {}).get('tags', [])

        # Print description and tags
        if primary_description:
            print("Image Description:", primary_description)
        if tags:
            print("Tags:", ", ".join(tags))

        if primary_description:
            poem = create_poem(primary_description, tags)
            print("\n" + poem)

        else:
            print("Couldn't retrieve a description for the image.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
