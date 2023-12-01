import json
import openai
import os
import re

os.system("cls")

# Define the conversation history file path
conversation_history_file = "conversation_history.json"
code_output_file = "Code Output.txt"
api_key_file = "api_key.json"
reset_frequency = 10

# Read the API key from the JSON file
try:
    with open(api_key_file, "r") as file:
        api_key_data = json.load(file)
        api_key = api_key_data["api_key"]
except (FileNotFoundError, KeyError):
    print(f"Unable to read the API key from {api_key_file}. Please ensure the file exists and contains a valid 'api_key' field.")
    exit(1)

# Set the API key
openai.api_key = api_key

# Load conversation history from file if it exists
try:
    with open(conversation_history_file, "r") as file:
        conversation_history = json.load(file)
except FileNotFoundError:
    conversation_history = []

while True:
    # Get user input
    user_input = input("You: ")

    # Break the loop if the user says 'quit'
    if user_input.lower() == "quit":
        break

    # Input validation
    if not user_input.strip():
        print("Please enter a valid input.")
        continue

    # Append the user input to the conversation history
    conversation_history.append({"role": "user", "content": user_input})

    # Check if the user wants to analyze a file
    if user_input.startswith("analyze file"):
        filename = user_input.replace("analyze file", "").strip()
        if not filename:
            print("Please provide a valid filename.")
            continue

        if not os.path.exists(filename):
            print(f"File '{filename}' does not exist.")
            continue

        try:
            with open(filename, "r", encoding="utf-8") as file:
                code_content = file.read()
        except UnicodeDecodeError:
            print(f"Unable to read the file '{filename}' due to an encoding issue.")
            continue

        # Prepend code content with appropriate instruction to Lyra
        conversation_history.append({"role": "assistant", "content": f"Please analyze the code file '{filename}':"})

        # Add code content to the conversation history
        conversation_history.append({"role": "user", "content": code_content})

    try:
        if len(conversation_history) > reset_frequency:
            conversation_history = conversation_history[-reset_frequency:]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation_history
        )

        response_text = response.choices[0].message.content.strip()

        print("Lyra: " + response_text)

        # Append the AI's response to the conversation history
        conversation_history.append({"role": "assistant", "content": response_text})

        # Save conversation history to file
        with open(conversation_history_file, "w") as file:
            json.dump(conversation_history, file)

        # Extract code from response
        code_matches = re.findall("```(.*?)```", response_text, re.DOTALL)
        extracted_code = "\n".join(code_matches)

        # Save extracted code to separate file
        with open(code_output_file, "w") as file:
            file.write(extracted_code)

    except Exception as e:
        print("Uh oh!:", str(e))
