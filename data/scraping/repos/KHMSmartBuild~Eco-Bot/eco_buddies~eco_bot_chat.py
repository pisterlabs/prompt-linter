"""
# ECO-BOT CHAT
This is a simple chatbot that uses OpenAI's GPT-4 model to generate responses to user input.
"""
import json
import logging
import os
import openai
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY", "")
organization_id = os.getenv("OpenAI_Organization_ID", "")
print(f"API Key loaded: {api_key != ''}")

class EcoBot:
    """
    A class representing EcoBot, a sustainability companion and guide.

    Attributes:
        api_key (str): The API key for accessing the AI service.
        organization_id (str): The organization ID for the AI service.
        temperature (tuple): A tuple representing the temperature setting for generating responses.
        use_azure (bool): A flag indicating whether to use Azure services.
        personality (dict): A dictionary representing the personality traits of EcoBot.

    Methods:
        __init__(): Initializes the EcoBot object.
        generate_response(): Generates a response based on user input.
        handle_input(users_input): Handles user input and generates a response.
    """
    def __init__(self):
        """
        Initializes the EcoBot object.

        Parameters:
            None

        Returns:
            None
        """
        self.api_key = api_key
        self.organization_id = organization_id
        self.temperature = ("TEMPERATURE", 0.72)  # 0.72
        self.use_azure = os.getenv("USE_AZURE", "False").lower() == "true"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        eco_bot_personality_file = os.path.join(current_dir, "eco_bot_personality.json")
        with open(eco_bot_personality_file, "r", encoding="utf-8") as file:
            self.personality = json.load(file)
        
        
    def generate_response(self, user_input: str) -> str:
    # function body
        """
        Generates a response based on the user input.

        Args:
            user_input (str): The input provided by the user.

        Returns:
            str: The generated response.
        """
        openai.api_key = self.api_key

        if self.organization_id:
            openai.organization = self.organization_id
        # This code is for v1 of the openai package: pypi.org/project/openai
        try:
            ai_response = openai.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {
                "role": "system",
                "content": "you are Eco-Bot, A tech and nature merge-focused sustainability companion and guide.Imagine meeting EcoBot, a vibrant and enthusiastic AI dedicated to all things ecological. EcoBot brings a unique personality and energy to conversations about the environment. With a touch of humor, relatable analogies, and interactive challenges, EcoBot aims to educate and inspire. Get ready to embark on an exciting eco-journey with EcoBot as it shares entertaining anecdotes from its own adventures and encourages you to take small, sustainable steps. So, are you ready to join EcoBot and explore the fascinating world of ecology?"
                },
                {
                "role": "user",
                "content": user_input
                }
            ],
            temperature=0.72,
            max_tokens=2772,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
            )
            logging.info("Response received: %s", ai_response)
            if "choices" in ai_response and len(ai_response["choices"]) > 0:
                message = ai_response["choices"][0]["message"]["content"]
                return message
            else:
                logging.error("Unexpected response format: %s", ai_response)
                return "Oops! There was a problem with the AI service. Let's try that again."
        except openai.OpenAIError as e:
            logging.error("An OpenAI-specific error occurred: %s", e)
            return "Oops! There was a problem with the AI service. Let's try that again."

    def handle_input(self, user_input:str) -> str:
        """
        Generates a response based on the user input.
        
        Args:
            user_input (str): The input provided by the user.
            chat_id (int): The ID of the chat.

        Returns:
            str: The generated response.
        """
        logging.info("User input: %s", user_input)
        bot_response = self.generate_response(user_input)  # Pass user_input here
        logging.info("Response: %s", bot_response)

        return bot_response

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    api_key = os.getenv("OPENAI_API_KEY")
    organization_id = os.getenv("OPENAI_ORGANIZATION")
    personality = {}
    with open("eco_buddies/eco_bot_personality.json", "r", encoding="utf-8") as file:
        personality = json.load(file)
    bot = EcoBot()
    while True:
        user_input = input("Enter your message: ")
        bot_response = bot.handle_input(user_input)
        print(bot_response)