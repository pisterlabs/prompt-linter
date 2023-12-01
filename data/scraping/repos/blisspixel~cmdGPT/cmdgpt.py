import os
import openai
import platform
import logging
import random
from dotenv import load_dotenv
from datetime import datetime
from colorama import init, Fore

load_dotenv()

init(autoreset=True)  # Initialize colorama

# Set up logging
if not os.path.exists('logs'):
    os.makedirs('logs')
log_filename = datetime.now().strftime("logs/cmdgptlog%Y%m%d.txt")
logging.basicConfig(filename=log_filename, level=logging.DEBUG)

openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = "2023-05-15"
openai.api_key = os.getenv("AZURE_OPENAI_KEY")

user_color = Fore.RED
cmdGPT_color = Fore.WHITE
system_color = Fore.LIGHTBLACK_EX

def clear_screen():
    if platform.system() == "Windows":
        os.system('cls')
    else:
        os.system('clear')

def display_initial_title():
    title_colors = [Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN, Fore.WHITE]
    chosen_color = random.choice(title_colors)
    title = f"""{chosen_color}
                    _  ____ ____ _____ 
  ___ _ __ ___   __| |/ ___|  _ \_   _|
 / __| '_ ` _ \ / _` | |  _| |_) || |  
| (__| | | | | | (_| | |_| |  __/ | |  
 \___|_| |_| |_|\__,_|\____|_|    |_|
    """
    print(title)
    print("\nInstructions:")
    print("- Type 'clear' to start a new chat.")
    print("- Type 'reset' to reset the chat and set a new system message.")
    print("- Type 'exit' or 'quit' to end the session.")
    print("----------------------------------------------")

def display_short_title(model_name):
    print(f"{system_color}cmdGPT | {model_name} | 'clear', 'reset', 'exit' or 'quit'")

def select_model():
    print("\nSelect a model:")
    print("1. gpt-35-turbo")
    print("2. gpt-35-turbo-16k")
    print("3. gpt-4")
    print("4. gpt-4-32k")
    print()
    choice = input("Enter your choice: ")
    models = {
        "1": "gpt-35-turbo",
        "2": "gpt-35-turbo-16k",
        "3": "gpt-4",
        "4": "gpt-4-32k"
    }
    return models.get(choice, "gpt-35-turbo")

def save_chat_transcript(messages):
    """
    Save the chat transcript to a file in the chat_transcripts directory.
    """
    try:
        if not os.path.exists('chat_transcripts'):
            os.makedirs('chat_transcripts')

        filename = datetime.now().strftime("chat_transcripts/chat_%Y%m%d%H%M%S.txt")

        with open(filename, 'w') as file:
            for message in messages:
                role = message["role"].capitalize()
                content = message["content"]
                file.write(f"{role}: {content}\n")

        manage_transcript_files()  # Ensure only the latest 10 transcript files are kept

    except Exception as e:
        logging.error("Error while saving chat transcript: %s", str(e))

def get_sorted_transcript_files():
    """Return a list of transcript files sorted by their creation time (oldest first)."""
    files = [os.path.join("chat_transcripts", f) for f in os.listdir("chat_transcripts")]
    files = [f for f in files if os.path.isfile(f)]
    return sorted(files, key=os.path.getctime)

def manage_transcript_files():
    """Keep only the latest 10 transcript files and remove the rest."""
    try:
        files = get_sorted_transcript_files()
        while len(files) > 10:  # Keep only the last 10 files
            os.remove(files.pop(0))  # Remove the oldest file

    except Exception as e:
        logging.error("Error while managing transcript files: %s", str(e))

def chat():
    while True:
        display_initial_title()
        model = select_model()
        clear_screen()
        display_short_title(model)

        messages = []

        system_message = input("\nEnter a system message or press Enter for default: ")
        if not system_message:
            system_message = "You are a helpful assistant."
        print(f"\n{system_color}System: {system_message}")
        messages.append({"role": "system", "content": system_message})

        while True:
            user_input = input(f"\n{user_color}You: ")
            if user_input.lower() in ["exit", "quit"]:
                save_chat_transcript(messages)  # Save transcript before exiting
                return
            elif user_input.lower() == "reset":
                save_chat_transcript(messages)  # Save transcript before resetting
                break
            elif user_input.lower() == "clear":
                save_chat_transcript(messages)  # Save transcript before clearing
                messages = [{"role": "system", "content": system_message}]  # Reset messages but retain the system message
                clear_screen()
                display_short_title(model)
                continue

            messages.append({"role": "user", "content": user_input})

            try:
                response = openai.ChatCompletion.create(engine=model, messages=messages)
                assistant_message = response['choices'][0]['message']['content']
                print(f"\n{cmdGPT_color}cmdGPT: {assistant_message}")
                messages.append({"role": "assistant", "content": assistant_message})
            except Exception as e:
                logging.error("Error while getting response from OpenAI: %s", str(e))
                print("There was an error processing the request. Check the logs for more details.")

if __name__ == "__main__":
    chat()
