import os
import openai
from dotenv import load_dotenv
import json
import subprocess

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


# Step 1: Define the main conversation logic
def run_conversation(user_input):
    # Step 2: Send user input to the GPT-3.5 Turbo model and specify available functions
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[{"role": "user", "content": user_input}],
        functions=[
            {
                "name": "get_windows_command_output",
                "description": "Run a Windows command and get the output",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The Windows command to run",
                        }
                    },
                    "required": ["command"],
                },
            }
        ],
        function_call="auto",
    )

    # Get the assistant's reply from the response
    message = response["choices"][0]["message"]

    # Step 3: Check if the model wants to call a function
    if message.get("function_call"):
        # Extract function name and arguments from the response
        function_name = message["function_call"]["name"]
        function_args = json.loads(message["function_call"]["arguments"])
        command = function_args.get("command")

        # Step 4: Prompt the user for confirmation (y/n)
        confirmation = input(f"The assistant wants to run the command: '{command}'. Do you approve? (y/n): ")

        if confirmation.lower() == "y":
            # Step 5: Execute the command if approved
            function_response = get_windows_command_output(command)
        else:
            # User declined the command, provide a response
            function_response = "Command execution declined by user."

        # Step 6: Continue the conversation by sending the function response to the model
        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[
                {"role": "user", "content": user_input},
                message,
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                },
            ],
        )
        return second_response


# Step 7: Define the function to execute a Windows command
def get_windows_command_output(command):
    try:
        # Execute the Windows command and capture the output
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output = result.stdout.strip()
        return output
    except Exception as e:
        return str(e)


# Step 8: Start the conversation loop
while True:
    # Prompt the user for input
    user_input = input("User: ")

    # Check if the user wants to exit
    if user_input.lower() == "exit":
        break

    # Step 9: Run the conversation logic with the user's input
    response = run_conversation(user_input)

    # Extract and print the assistant's response
    assistant_response = response["choices"][0]["message"]["content"]
    print("Assistant:", assistant_response)
