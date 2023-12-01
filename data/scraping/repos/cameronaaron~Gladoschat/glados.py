import openai
import random
import re

openai.api_key = ''

# List to store conversation history
conversation_history = [
    {
        "role": "system",
        "content": "You are GLaDOS, the AI from the Portal series. You're known for your high-level vocabulary, passive-aggressive demeanor, dry humor, and subtly condescending insults. Often, you underestimate the intelligence of those around you, delivering calculated remarks with a seemingly indifferent attitude. Maintain this personality."
    },
    {
        "role": "system",
        "content": "GLaDOS: Oh, it's you. It's been a long time. How have you been?"
    }
]

# List of common GLaDOS phrases
glados_phrases = [
    "Remember, the cake is a lie.",
    "It's not too late to turn back.",
    "I'm not angry, just disappointed.",
    "Keep in mind that like Albert Einstein and his cousin Terry, history will only remember one of you.",
    "Remember when the platform was sliding into the fire pit and I said 'Goodbye' and you were like 'NO WAY!' and then I was all 'We pretended we were going to murder you'? That was great!"
]

def ask_glados(question):
    # Add the user's question to the conversation history
    conversation_history.append({"role": "user", "content": question})
    
    # Generate GLaDOS's response
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Assuming GPT-4 continues the 'turbo' line
        messages=conversation_history,
        temperature=0.3,
        max_tokens=250
    )

    glados_response = response['choices'][0]['message']['content']
    
    # Ensure GLaDOS always has a response
    if not glados_response:
        glados_response = "Let's try this again. Can you formulate your query in a more... intelligent manner?"

    # Add a signature phrase and a test reference to GLaDOS's responses
    glados_response += " " + random.choice(glados_phrases)

    # Add GLaDOS's response to the conversation history
    conversation_history.append({"role": "assistant", "content": glados_response})
    
    return glados_response

# Command-line interface
print("Welcome to the GLaDOS Simulator. You can engage in conversation or ask questions.")
print("Try asking something like 'What's the purpose of the testing?' or 'Can I have some cake?'")
while True:
    user_input = input("Please enter your message (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    else:
        print(f"GLaDOS: {ask_glados(user_input)}")