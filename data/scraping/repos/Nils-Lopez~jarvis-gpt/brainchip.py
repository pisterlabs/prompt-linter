import openai
import os

from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key using the value from the environment variables
openai.api_key = os.getenv('OPEN_AI_APIKEY')

# Define a function to interact with the GPT-3 model
def ask_gpt(new_sentence):
    # Create a chat-based completion using GPT-3.5 Turbo
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            # System message to set the behavior of the assistant (in this case, Jarvis-like behavior)
            {"role": "system", "content": "Respond to this question as if you were Jarvis, a futuristic personal assistant, reply in under 20 words, always with geeky comments, end each sentence with a geek culture reference"},
            
            # User message with the input sentence or question
            {"role": "user", "content": new_sentence}
        ])

    # Extract the response from GPT-3 and return it along with a "False" flag indicating no error
    return response.choices[0].message.content, False
