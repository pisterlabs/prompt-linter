# Load environment variables
from dotenv import load_dotenv; load_dotenv()

# Fetch environment variables
import os

# OpenAI Python SDK to allow us to make requests to GPT
import openai

# ===========================
# Welcome! This is the part of the file where you'll be writing your code.
# Feel free to add to it and experiment, but altering the starter code is not recommended.
# Make sure to read the comments carefully!
# Don't hesitate to ask for help if you're stuck. Happy coding!
# ===========================

# ===========================
# === MODIFY THIS SECTION ===
# ===========================

# Set OpenAI variables as shown in documenation
openai.api_key = os.getenv('OPENAI_API_KEY')

# Call Azure to send the message to GPT and return the response string
def sendMessage(text: str) -> str:
    raise Exception("Exercise 1: send message incomplete.") # Remove this line when you start working on this function
