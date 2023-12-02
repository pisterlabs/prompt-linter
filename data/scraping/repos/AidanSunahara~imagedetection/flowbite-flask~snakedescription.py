import openai
import os
from dotenv import load_dotenv

def snakeDescription(snake):   
    load_dotenv()
    # Replace 'your-api-key' with your actual OpenAI API key
    apiKey = os.getenv("api_key")

    openai.api_key = apiKey

    # Define the conversation for the chat-based API
    conversation = [
        {"role": "user", "content": f"Generate a summary of the {snake}. 2 sentences."},
    ]


    try: 
    # Make a request to the OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation,
        )
    except Exception as e:
    
        print(f'An error occurred: {e}')
    
    
    return response['choices'][0]['message']['content']






