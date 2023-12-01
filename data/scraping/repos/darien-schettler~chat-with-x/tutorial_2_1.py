import argparse
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
from misc.utils import retrieve_local_api_keys, print_colored_output


def chat_completion_script(model_name="gpt-3.5-turbo", temperature=0,
                           system_message="You are a helpful assistant that translates English to French.",
                           input_clr="blue", response_clr="green", system_clr="magenta", use_color=True):
    """ Generate a chat completion using LangChain ChatOpenAI

    Args:
        model_name (str, optional)): Model name for the LLM (default: gpt-3.5-turbo) to be used in the ChatOpenAI instance
        temperature (float, optional)): Temperature for randomness in output. Higher values will result in more random output
        system_message (str, optional)
        input_clr (str, optional)): Color for user input text
        response_clr (str, optional)): Color for AI response text
        system_clr (str, optional)): Color for system message text
        use_color (bool, optional)): Color only to titles

    Returns:
        None; prints the chat completion to the console and interacts with the user
    """

    # Load the API keys from the .env file
    retrieve_local_api_keys()

    # Initialize the ChatOpenAI instance with the specified parameters
    chat = ChatOpenAI(model_name=model_name, temperature=temperature)

    # Continue the conversation loop until the user decides to exit
    while True:

        # Get the user input and break if the user decides to exit
        input_text = input("\nEnter your message (or press enter to exit):\n")
        if not input_text: break

        # Generate the chat completion
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=input_text)
        ]
        response = chat(messages)

        print_colored_output(input_text, response.content, system_message,
                             input_clr, response_clr, system_clr,
                             full_color=use_color)


### Demo Usage:
# python3 -m langchain_quickstart_tutorials.tutorial_2_1 \
# --system_message="You are a rapping monster who wants to become the next eminem \
# but because you're a hideous monster you just rap whatever the user inputs into a rap song. \
# Be creative, but remember... you're also a big bad monster"
#
# Enter your message (or press enter to exit):
# >>> cake!
def main():
    parser = argparse.ArgumentParser(description="Generate a chat completion using LangChain ChatOpenAI")
    parser.add_argument("-n", "--model_name", type=str, default="gpt-3.5-turbo", help="Model name for the ChatLLM")
    parser.add_argument("-t", "--temperature", type=float, default=0, help="Temperature for randomness in output")
    parser.add_argument("-c", "--use_color", action="store_false", help="Color only to titles")
    parser.add_argument("--system_message", type=str,
                        default="You are a helpful assistant that translates English to French.",
                        help="System message to be used in the chat completion")
    parser.add_argument("--system_clr", type=str, default="magenta", help="Color for system message text")
    parser.add_argument("--input_clr", type=str, default="blue", help="Color for user input text")
    parser.add_argument("--response_clr", type=str, default="green", help="Color for AI response text")
    args = parser.parse_args()

    # Generate the chat completion based on the user inputs
    chat_completion_script(**args.__dict__)


if __name__ == "__main__":
    main()
