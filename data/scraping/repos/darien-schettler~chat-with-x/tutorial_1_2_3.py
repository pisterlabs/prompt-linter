import argparse
from misc.utils import print_colored_output
from langchain.llms.fake import FakeListLLM
from langchain.agents import load_tools
from langchain.agents import initialize_agent


def llm_interaction(
        tool_name="python_repl",
        responses = ("Action: Python REPL\nAction Input: print(2 + 2)", "Final Answer: 4"),
        agent="zero-shot-react-description", verbose=True, **kwargs
):
    """ Script to interact with a 'Fake' LLM to mock responses from a real LLM

    Args:


    Returns:
        None; interacts with the user and prints the user inputs and model responses to the console
    """

    tools = load_tools([tool_name])
    llm = FakeListLLM(responses=responses)
    agent = initialize_agent(tools, llm, agent, verbose=verbose, **kwargs)

    # Continue the conversation loop until the user decides to exit
    while True:

        # Get the user input and break if the user decides to exit
        input_text = input("\nEnter your message - i.e. 'What's 2+2?' (or press enter to exit):\n")
        if not input_text: break

        response = agent.run(input_text)
        print_colored_output(input_text, response_text=response.strip(), full_color=True)


def main():
    parser = argparse.ArgumentParser(description="Interact with a fake LLM model via LangChain to mock responses")
    parser.add_argument("-tn", "--tool_name", type=str, default="python_repl",
                        help="Name of the tool used for interaction.")
    parser.add_argument("-r", "--responses", nargs=2,
                        default=("Action: Python REPL\nAction Input: print(2 + 2)", "Final Answer: 4"),
                        help="Tuple containing the responses to be mocked.")
    parser.add_argument("-a", "--agent", type=str, default="zero-shot-react-description",
                        help="Name of the agent used for interaction.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Flag to set the verbosity of the output.")
    args = parser.parse_args()

    # Call the main function
    llm_interaction(**args.__dict__)


if __name__ == "__main__":
    main()

