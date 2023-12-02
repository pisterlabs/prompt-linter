import ast
import argparse
from langchain.llms import OpenAI
from misc.utils import retrieve_local_api_keys, print_colored_output, flatten_l_o_l


def llm_interaction(model_name="text-ada-001", temperature=0.75, repeats=1, best_of=1,
                    input_clr="blue", response_clr="red", use_color=True):
    """ Script to interact with an OpenAI LLM model

    Args:
        model_name (str, optional): Model name for the LLM to be used in the ChatOpenAI instance
        temperature (float, optional): Temperature for randomness in output. Higher values will result in more random
        repeats (int, optional): How many completions to generate for each prompt.
        best_of (int, optional): Generates completions server-side and returns the best
        input_clr (str, optional): Color for user input text
        response_clr (str, optional): Color for AI response text
        use_color (bool, optional): Color only to titles

    Returns:
        None; interacts with the user and prints the user inputs and model responses to the console
    """

    # Initialize the ChatOpenAI instance with the specified parameters
    llm = OpenAI(model_name=model_name, temperature=temperature, n=repeats, best_of=best_of)

    # Continue the conversation loop until the user decides to exit
    while True:

        # Get the user input and break if the user decides to exit
        input_text = input("\nEnter your message (or press enter to exit):\n")
        if not input_text: break

        if input_text[0] == "[":
            input_text = ast.literal_eval(input_text)
            responses = flatten_l_o_l([[_x.text for _x in x] for x in llm.generate(input_text).generations])
        else:
            responses = [llm(input_text),]

        for r in responses:
            print_colored_output(
                input_text, response_text=r.strip(), input_color=input_clr, response_color=response_clr,
                full_color=use_color
            )


def main():
    parser = argparse.ArgumentParser(description="Generate a LLM model via LangChain OpenAI module")
    parser.add_argument("-n", "--model_name", type=str, default="text-ada-001", help="Model name for the LLM")
    parser.add_argument("-t", "--temperature", type=float, default=0.75, help="Temperature for randomness in output")
    parser.add_argument("--repeats", type=int, default=1, help="How many completions to generate for each prompt.")
    parser.add_argument("--best_of", type=int, default=1, help="Generates completions server-side and returns the best")
    parser.add_argument("-c", "--use_color", action="store_false", help="Color only to titles")
    parser.add_argument("--input_clr", type=str, default="blue", help="Color for user input text")
    parser.add_argument("--response_clr", type=str, default="red", help="Color for AI response text")
    args = parser.parse_args()

    # Load the API keys from the .env file
    retrieve_local_api_keys()

    # Call the main function
    llm_interaction(**args.__dict__)


if __name__ == "__main__":
    main()
