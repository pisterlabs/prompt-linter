# Import the necessary libraries
import openai   # for the OpenAI API
import os       # to interact with the operating system
import requests # to send HTTP requests
import datetime # to work with dates and times
import base64   # to encode and decode binary data
import json     # to work with JSON data

# This function opens a file and reads its contents
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# This function writes content to a file
def save_file(filepath, content):
    with open(filepath, 'a', encoding='utf-8') as outfile:
        outfile.write(content)

# Here, we read API keys from text files
api_key = open_file('openaiapikey2.txt')  # OpenAI API key
sd_api_key = open_file('sdapikey.txt')    # Stability.AI API key

# This function calls the OpenAI API's ChatCompletion endpoint to carry out a conversation
def chatgpt(api_key, conversation, chatbot, user_input, temperature=0.8, frequency_penalty=0.2, presence_penalty=0):
    openai.api_key = api_key

    # Add user input to the conversation history
    conversation.append({"role": "user","content": user_input})

    # Add a system message to the conversation
    messages_input = conversation.copy()
    prompt = [{"role": "system", "content": chatbot}]
    messages_input.insert(0, prompt[0])

    # Make a call to the OpenAI API
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        messages=messages_input)

    # Extract ChatGPTs response
    chat_response = completion['choices'][0]['message']['content']

    # Add ChatGPTs response to the conversation
    conversation.append({"role": "assistant", "content": chat_response})

    # Return ChatGPTs response
    return chat_response

# This function calls the Stability.AI API to generate an image from text prompts
def generate_image(api_key, text_prompt, negative_prompts, height=512, width=512, cfg_scale=9, clip_guidance_preset="FAST_BLUE", steps=70, samples=1):
    api_host = 'https://api.stability.ai'
    engine_id = "stable-diffusion-xl-beta-v2-2-2"

    # Here, we're making a POST request to the SD API with our parameters
    response = requests.post(
        f"{api_host}/v1/generation/{engine_id}/text-to-image",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        json={
            "text_prompts": [
                {
                    "text": text_prompt # Our Text Prompt
                }
            ],
            "negative_prompts": [
                {
                    "text": negative_prompts # Our Negative Prompt
                }
            ],
            "cfg_scale": cfg_scale,
            "clip_guidance_preset": clip_guidance_preset,
            "height": height,
            "width": width,
            "samples": samples,
            "steps": steps,
        },
    )

    # Check for errors in the API response
    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    # Extract the base64 image data from the response
    data = response.json()
    image_data = data["artifacts"][0]["base64"]

    # Save the image data as a PNG file
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    image_filename = os.path.join("SDimages", f"generated_image_{timestamp}.png") # Where we save your image "SDimages" folder

    with open(image_filename, "wb") as f:
        f.write(base64.b64decode(image_data))

    # Return the file path of the saved image
    return image_filename   

# Start with an empty conversation
conversation = []

# Here we start the main loop, n is how many images we care gonna create
n = 10 
for i in range(n):
    # Reset conversation for each loop
    conversation = []

    # Define an system message for ChatGPT
    chatbot = "You are Photograph and Art Expert. You task to creating amazing prompts"

    # Read prompts from a text file and use the OpenAI assistant to refine them
    text_prompt1 = open_file("prompts.txt")
    text_prompt = chatgpt(api_key, conversation, chatbot, text_prompt1)
    print(text_prompt)
    negative_prompts1 = open_file("nprompts.txt")
    negative_prompts = chatgpt(api_key, conversation, chatbot, negative_prompts1)
    print(negative_prompts)

    # Call the Stability.AI API to generate an image using the refined prompts
    image_filename = generate_image(sd_api_key, text_prompt, negative_prompts)
