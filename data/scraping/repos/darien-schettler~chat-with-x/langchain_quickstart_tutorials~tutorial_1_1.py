import argparse
from langchain.llms import OpenAI
from colorama import init, Fore, Back, Style
from misc.utils import retrieve_local_api_keys


def generate_text(input_text, model_name='text-davinci-003', n_repeats=1, temperature=0.7):
    """ Generate text using the LangChain LLM

    Args:
        input_text (str): Input text for the LLM
        model_name (str, optional): Model name for the LLM
        n_repeats (int, optional): Number of responses
        temperature (float, optional): Temperature for randomness in output

    Returns:
        None; prints the generated text
    """
    # Initialize the LLM wrapper with the specified parameters
    llm = OpenAI(model_name=model_name, n=n_repeats, temperature=temperature)

    # Call the LLM on the input text and print the result
    # Call the LLM on the input text and print the result
    print(f"\n{Fore.BLUE}Input Text    :{Fore.RESET} '{input_text}'")
    print(f"{Fore.GREEN}Response Text :{Fore.RESET} '{llm(input_text).strip()}'\n")


def main():
    """ Execute the main script to generate text using LangChain LLM """
    parser = argparse.ArgumentParser(description="Generate text using LangChain LLM")
    parser.add_argument("input_text", type=str, help="Input text for the LLM")
    parser.add_argument("-n", "--model_name", type=str, default="text-davinci-003", help="Model name for the LLM")
    parser.add_argument("-r", "--repeats", type=int, default=1, help="Number of responses")
    parser.add_argument("-t", "--temperature", type=float, default=0.7, help="Temperature for randomness in output")
    args = parser.parse_args()

    # Load the API keys from the .env file
    retrieve_local_api_keys()

    # Generate the text based on the user inputs
    generate_text(args.input_text, model_name=args.model_name, n_repeats=args.repeats, temperature=args.temperature)


if __name__ == "__main__":
    main()
