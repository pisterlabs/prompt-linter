from openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv

BASEDIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(BASEDIR, '.env'))

client = OpenAI(
   api_key=os.getenv('OPENAI_API_KEY'),
)

# Assuming you have your conversation history defined somewhere, you can import it here
# If it's in your main app file, you may need to find a way to pass it to the function or redesign the code structure
# from app import conversation_history

def chat_with_gpt(question, conversation_history, personality):
    """
    Function to interact with ChatGPT.

    Args:
    - question (str): The question to ask ChatGPT.
    - conversation_history (list): List containing conversation history.

    Returns:
    - str: Answer from ChatGPT.
    """

    # Ensure conversation history doesn't exceed the token limit (e.g., 4096 tokens)
    while sum(len(message['content'].split()) for message in conversation_history) > 4096:
        conversation_history.pop(0)  # Remove the oldest message

    # Create a message with the user's question
    user_message = {"role": "user", "content": question}

    # Add the user's message to the conversation history
    conversation_history.append(user_message)

    # Use OpenAI to get an answer
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
             {"role": "system", "content": personality},
              *conversation_history,
            ]
    )

    # Extract the answer from the response
    answer = response.choices[0].message.content

    # Add the model's response to the chat windows conversation history
    conversation_history.append({"role": "assistant", "content": answer})

    return answer