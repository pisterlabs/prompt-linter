import time
import asyncio
import argparse
from langchain.llms import OpenAI
from misc.utils import retrieve_local_api_keys, print_colored_output, timeit


@timeit
def generate_serially(llm, input_text, n_reps=3):
    for _ in range(n_reps):
        resp = llm.generate([input_text])
        print_colored_output(input_text, response_text=resp.generations[0][0].text.strip(), full_color=True)


async def async_generate(llm, input_text):
    resp = await llm.agenerate([input_text])
    print_colored_output(input_text, response_text=resp.generations[0][0].text.strip(), full_color=True)


@timeit
async def generate_concurrently(llm, input_text, n_reps=3):
    tasks = [async_generate(llm, input_text) for _ in range(n_reps)]
    await asyncio.gather(*tasks)


def llm_interaction(model_name="text-ada-001", temperature=0.75, n_reps=3, method="serial"):
    """ Script to interact with an OpenAI LLM model

    Args:
        model_name (str, optional): Model name for the LLM to be used in the ChatOpenAI instance
        temperature (float, optional): Temperature for randomness in output. Higher values will result in more random
        n_reps (int, optional): How many completions to generate for each prompt.
        method (str, optional): How to generate the responses. Options: serial, concurrent, both

    Returns:
        None; interacts with the user and prints the user inputs and model responses to the console
    """

    # Initialize the ChatOpenAI instance with the specified parameters
    llm = OpenAI(model_name=model_name, temperature=temperature)

    # Continue the conversation loop until the user decides to exit
    while True:

        # Get the user input and break if the user decides to exit
        input_text = input("\nEnter your message (or press enter to exit):\n")
        if not input_text: break

        # Generate the response
        if method == "serial":
            generate_serially(llm, input_text, n_reps=n_reps)
        elif method == "concurrent":
            # If running this inside Jupyter, use await generate_concurrently(llm, input_text, n_reps)
            asyncio.run(generate_concurrently(llm, input_text, n_reps))
        elif method == "both":
            generate_serially(llm, input_text, n_reps=n_reps)
            asyncio.run(generate_concurrently(llm, input_text, n_reps))
        else:
            raise ValueError(f"Method {method} not recognized. Please use 'serial' or 'concurrent'.")


def main():
    parser = argparse.ArgumentParser(description="Generate a LLM model via LangChain OpenAI module")
    parser.add_argument("-n", "--model_name", type=str, default="text-ada-001", help="Model name for the LLM")
    parser.add_argument("-t", "--temperature", type=float, default=0.75, help="Temperature for randomness in output")
    parser.add_argument("--n_reps", type=int, default=3, help="How many times to generate for each prompt.")
    parser.add_argument("--method", type=str, default="serial", help="Serial or concurrent generation.")
    args = parser.parse_args()

    # Load the API keys from the .env file
    retrieve_local_api_keys()

    # Call the main function
    llm_interaction(**args.__dict__)


if __name__ == "__main__":
    main()
