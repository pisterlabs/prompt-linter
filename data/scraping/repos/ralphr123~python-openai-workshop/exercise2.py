# Load environment variables from .env
from dotenv import load_dotenv; load_dotenv()

# OpenAI Python SDK to allow us to make requests to GPT
import openai

# ===========================
# === MODIFY THIS SECTION ===
# ===========================

# List to store message history
history = []

# Add a message of role "user" to the history array
def addUserMessage(text: str):
	history.append({ 'role': 'user', 'content': text })

# Add a message of role "assistant" to the history array
def addAIMessage(text: str):
	raise NotImplementedError('addAIMessage() has not been implemented yet') # Remove this line when you start working on this function

# Send a message to GPT and return the response string
def sendMessage(text: str) -> str:
	# 1. Use the function you completed to add the message as a user message

	# 2. Call Azure to send the message to GPT (similar to exercise 1)

	# 3. Store the result in a variable

	# 4. Use the function you completed to add the response as an AI message
	
	# 5. Return the response string

	raise NotImplementedError('sendMessage() has not been implemented yet') # Remove this line when you start working on this function