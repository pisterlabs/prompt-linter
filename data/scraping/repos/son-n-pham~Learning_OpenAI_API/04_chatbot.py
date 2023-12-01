import openai
from dotenv import dotenv_values
import argparse

# Load OpenAI API key from .env file
config = dotenv_values(".env")
openai.api_key = config["OPENAI_API_KEY"]

# Define functions to print colored text
def print_bold(text):
    return f"\033[1m{text}\033[0m"

def print_red(text):
    return f"\033[31m{text}\033[0m"

def print_blue(text):
    return f"\033[34m{text}\033[0m"

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Simple command line chatbot')
    parser.add_argument("--personality", type=str, default="friendly helpful", help="Personality of the chatbot")
    args = parser.parse_args()

    # Print command line arguments
    print(f"Starting chatbot with personality: {args.personality}")

    # Initialize chatbot messages
    initial_prompt = f"You are a conversational chatbot. Your personality is {args.personality}"
    messages = [{"role": "system", "content": initial_prompt}]

    # Start chatbot loop
    while True:
        try:
            # Get user input
            user_input = input(f"{print_blue(print_bold('You:'))} ")
            messages.append({"role": "user", "content": user_input})

            # Get chatbot response from OpenAI API
            res = openai.ChatCompletion.create(
                model="davinci",
                messages=messages)

            bot_answer = res.choices[0].text
            print(f"{print_red(print_bold('Bot:'))} {bot_answer}\n")
            messages.append({"role": "assistant", "content": bot_answer})

        except (KeyboardInterrupt, EOFError):
            # Handle keyboard interrupt or end of file
            print("\nExiting...")
            break

        except Exception as e:
            # Handle other exceptions
            print(f"Error: {e}")

if __name__ == "__main__":
    main()