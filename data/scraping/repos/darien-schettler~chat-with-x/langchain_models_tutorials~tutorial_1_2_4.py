import argparse
import langchain
from misc.utils import print_colored_output, retrieve_local_api_keys
from misc.caching_utils import create_cache
from misc.utils import TimedOpenAI


def llm_interaction(cache_method="in_memory", db_path=None, remove_existing_db=True, engine_path=None,
                    model_name="text-davinci-002", **kwargs):
    """ Script to interact with an LLM to investigate caching behavior and performance

    Args:


    Returns:
        None; interacts with the user and prints the user inputs and model responses to the console
    """

    if cache_method is not None:
        langchain.llm_cache = create_cache(
            cache_method, db_path=db_path, remove_existing_db=remove_existing_db, engine_path=engine_path
        )
        do_cache = True
    else:
        do_cache = False

    llm = TimedOpenAI(model_name=model_name, cache=do_cache)

    # Continue the conversation loop until the user decides to exit
    while True:

        # Get the user input and break if the user decides to exit
        input_text = input("\nEnter your message (or press enter to exit):\n")
        if not input_text: break

        response = llm(input_text)
        print_colored_output(input_text, response_text=response.strip(), full_color=True)


def main():
    parser = argparse.ArgumentParser(description="Interact with a fake LLM model via LangChain to mock responses")
    parser.add_argument("-n", "--model_name", type=str, default="text-davinci-002", help="Model name for the LLM")
    parser.add_argument("-c", "--cache_method", type=str, default="in_memory",
                        help="What type of caching to use. Options are: in_memory, sqlite, redis, sqlalchemy, or None")
    parser.add_argument("-v", "--verbose", action="store_true", help="Flag to set the verbosity of the output.")
    parser.add_argument("-d", "--db_path", type=str, default=None, help="Path to the database file")
    parser.add_argument("-r", "--remove_existing_db", action="store_true", help="Remove existing database if set")
    parser.add_argument("-e", "--engine_path", type=str, default=None, help="Path to the engine")

    args = parser.parse_args()
    print(args.__dict__)
    # Load the API keys from the .env file
    retrieve_local_api_keys()

    # Call the main function
    llm_interaction(**args.__dict__)


if __name__ == "__main__":
    main()

