import argparse
from langchain.agents import load_tools, initialize_agent
from langchain.llms import OpenAI
from misc.utils import retrieve_local_api_keys, print_colored_output


def agent_script(query,
                 model_name="text-davinci-003",
                 temperature=0.0,
                 agent_type="zero-shot-react-description",
                 verbose=True,
                 input_clr="blue",
                 response_clr="green",
                 **kwargs):
    """ Run a query using a LangChain agent

    Args:
        query (str): The query to be run by the agent
        model_name (str): Model name for the LLM
        temperature (float): Temperature for randomness in output
        agent_type (str): Type of agent to use
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

    # Load the required tools
    tools = load_tools(["google-serper", "llm-math"], llm=llm)

    # Initialize the agent
    agent = initialize_agent(tools, llm, agent=agent_type, verbose=verbose)

    # Run the agent with the given query
    response = agent.run(query)

    # Print the response
    print_colored_output(query, response, input_clr, response_clr)


def main():
    parser = argparse.ArgumentParser(description="Run a query using a LangChain agent")
    parser.add_argument("query", type=str, help="The query to be run by the agent")
    parser.add_argument("-n", "--model_name", type=str, default="text-davinci-003", help="Model name for the LLM")
    parser.add_argument("-t", "--temperature", type=float, default=0.0, help="Temperature for randomness in output")
    parser.add_argument("-a", "--agent_type", type=str, default="zero-shot-react-description",
                        help="Type of agent to use")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--input_clr", type=str, default="blue", help="Color to use for input text label")
    parser.add_argument("--response_clr", type=str, default="green", help="Color to use for response text label")
    args = parser.parse_args()

    # Run the agent script with the provided arguments
    agent_script(**args.__dict__)


# Example Usage:
# python3 -m langchain_quickstart_tutorials.tutorial_1_3 \
# "List all AI models that were released in 2023 as well as who released them and the date they were released:" -v
if __name__ == "__main__":
    main()
