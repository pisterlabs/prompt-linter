import argparse
from langchain import OpenAI, ConversationChain
from misc.utils import retrieve_local_api_keys, print_colored_output


def conversation_script(model_name="text-davinci-003", temperature=0, verbose=True,
                        input_clr="blue", response_clr="green", **kwargs):
    """ Run a conversation using a LangChain ConversationChain

    Args:
        model_name (str): Model name for the LLM
        temperature (float): Temperature for randomness in output. A value of 0 will result in the most likely output.
        verbose (bool): Verbose output
        input_clr (str): Color to use for input text label
        response_clr (str): Color to use for response text label
        **kwargs: Additional keyword arguments

    Returns:
        None; prints the response to the query
    """
    # Load the API keys from the .env file
    retrieve_local_api_keys()

    # Initialize the LLM with the specified parameters
    llm = OpenAI(model_name=model_name, temperature=temperature)

    # Initialize the ConversationChain
    conversation = ConversationChain(llm=llm, verbose=verbose)

    # Continue the conversation loop until the user decides to exit
    while True:
        input_text = input("\nEnter your message (or press enter to exit):\n")
        if not input_text: break
        response = conversation.predict(input=input_text)
        print_colored_output(input_text, response, input_clr, response_clr)


def main():
    parser = argparse.ArgumentParser(description="Run a conversation using a LangChain ConversationChain")
    parser.add_argument("-n", "--model_name", type=str, default="text-davinci-003", help="Model name for the LLM")
    parser.add_argument("-t", "--temperature", type=float, default=0, help="Temperature for randomness in output")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--input_clr", type=str, default="blue", help="Color to use for input text label")
    parser.add_argument("--response_clr", type=str, default="green", help="Color to use for response text label")
    args = parser.parse_args()

    # Run the conversation script with the provided arguments
    conversation_script(**args.__dict__)


if __name__ == "__main__":
    main()
