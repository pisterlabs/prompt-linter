import openai
import json

# Set up your OpenAI API credentials
openai.api_key = 'YOUR_API_KEY'

# Define a function to send a message to the chatbot and get a response
def send_message(message, history=[]):
    input_prompt = '\n'.join(history + [message])
    
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=input_prompt,
        max_tokens=50,
        temperature=0.7,
        n=1,
        stop=None,
        timeout=10
    )

    return response.choices[0].text.strip()

# Main loop for chatting with the bot
def chat_with_bot():
    print("Welcome to the ChatGPT bot! Type 'exit' to end the conversation.")
    message = input("You: ")
    history = []

    while message.lower() != 'exit':
        # Validate and preprocess user input
        if not message:
            print("Please enter a message.")
            message = input("You: ")
            continue

        # Send the message to the bot and get the response
        response = send_message(message, history)
        print("ChatGPT: " + response)

        # Add the user message and bot response to the conversation history
        history.append(message)
        history.append(response)

        # Get the next user message
        message = input("You: ")

# Start the conversation
chat_with_bot()
