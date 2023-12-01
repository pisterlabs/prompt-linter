import argparse
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI

from misc.utils import retrieve_local_api_keys, print_colored_output


def chat_completion_script(model_name="gpt-3.5-turbo", temperature=0,
                           agent_type="zero-shot-react-description",
                           input_clr="blue", response_clr="green",
                           use_color=True, verbose=False, **kwargs):
    """ Generate a chat completion using LangChain ChatOpenAI

    Args:
        model_name (str, optional)): Model name for the Chatbot (default: gpt-3.5-turbo)
        temperature (float, optional)): Temperature for randomness in output. A value of 0 means no randomness.
        agent_type (str, optional)): Type of agent to use (default: zero-shot-react-description)
        input_clr (str, optional)): Color for user input text
        response_clr (str, optional)): Color for AI response text
        use_color (bool, optional)): Color only to titles
        verbose (bool, optional)): Agent verbosity (default: False)
        **kwargs: Additional keyword arguments

    Returns:
        None; prints the response to the query using the specified agent
    """

    # Load the API keys from the .env file
    retrieve_local_api_keys()

    # First, let's load the language model we're going to use to control the agent.
    chat = ChatOpenAI(model_name=model_name, temperature=temperature)

    # Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
    #   --> Note we don't pass a model name so it will default to davinici-003
    llm = OpenAI(temperature=temperature)
    tools = load_tools(["google-serper", "llm-math"], llm=llm)

    # Initialize the agent
    agent = initialize_agent(tools, chat, agent=agent_type, verbose=verbose)

    # Continue the conversation loop until the user decides to exit
    while True:
        input_text = input("\nEnter your message (or press enter to exit):\n")
        if not input_text: break

        # Run the agent with the given query
        response = agent.run(input_text)

        # Print the response
        print_colored_output(input_text, response,
                             input_color=input_clr,
                             response_color=response_clr,
                             full_color=use_color)


# Example Usage:
# python3 -m langchain_quickstart_tutorials.tutorial_2_4 -v
#
# Enter your message (or press enter to exit):
# >>> 'Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?'
def main():
    parser = argparse.ArgumentParser(description="Generate a chat completion using LangChain ChatOpenAI")
    parser.add_argument("-n", "--model_name", type=str, default="gpt-3.5-turbo", help="Model name for the ChatLLM")
    parser.add_argument("-t", "--temperature", type=float, default=0, help="Temperature for randomness in output")
    parser.add_argument("-c", "--use_color", action="store_false", help="Color only to titles")
    parser.add_argument("-v", "--verbose", action="store_true", help="Agent verbosity (default: False)")
    parser.add_argument("--input_clr", type=str, default="blue", help="Color for user input text")
    parser.add_argument("--response_clr", type=str, default="green", help="Color for AI response text")
    args = parser.parse_args()

    # Generate the chat completion based on the user inputs
    chat_completion_script(**args.__dict__)


if __name__ == "__main__":
    main()
