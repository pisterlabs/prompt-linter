import argparse
from misc.utils import print_colored_output, retrieve_local_api_keys, get_streaming_cb, simple_colorizer
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.callbacks import get_openai_callback
from langchain.agents import load_tools
from langchain.agents import initialize_agent



def llm_interaction(model_name="text-davinci-002", temperature=0.0, use_color=True, input_clr="blue",
                    response_clr="red", verbose=True, streaming=False, use_agent=True, **kwargs):
    """ Interact with an LLM to demonstrate the streaming functionality within LangChain

    Args:
        model_name (str, optional): Model name for the LLM (default: text-davinci-002) to be used
        temperature (float, optional): Temperature for randomness in output
        use_color (bool, optional): Color only to titles
        input_clr (str, optional): Color for user input text
        response_clr (str, optional): Color for AI response text
        verbose (bool, optional): Flag to set the verbosity of the output.
        streaming (bool, optional): Flag to add streaming functionality.
        use_agent (bool, optional): Flag to add agent functionality.

    Returns:
        None; interacts with the user and prints the user inputs and model responses to the console
    """
    agent = None
    use_chat = True if "gpt" in model_name else False
    if use_chat:
        chat = ChatOpenAI(
            model_name=model_name, temperature=temperature,
            streaming=streaming, verbose=verbose,
            # callback_manager=get_streaming_cb()
        )
        llm = None
    else:
        chat = None
        llm = OpenAI(
            model_name=model_name, temperature=temperature,
            streaming=streaming, verbose=verbose,
            # callback_manager=get_streaming_cb()
        )

    if use_agent:
        if chat is None:
            print("\n... Model passed is not a chat model. Using default chat model ...\n")
            chat = ChatOpenAI(
                temperature=temperature, streaming=streaming, verbose=verbose, callback_manager=get_streaming_cb()
            )
            llm = OpenAI(
                model_name=model_name, temperature=0,
            )
        else:
            llm = OpenAI(temperature=0)

        # Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
        #   --> Note we don't pass a model name so it will default to davinici-003
        tools = load_tools(["google-serper", "llm-math"], llm=llm)

        # Initialize the agent
        agent = initialize_agent(tools, chat, agent="zero-shot-react-description", verbose=verbose)

    # Continue the conversation loop until the user decides to exit
    with get_openai_callback() as cb:
        while True:
            # Get the user input and break if the user decides to exit
            input_text = input("\nEnter your message (or press enter to exit):\n")
            if not input_text: break

            if use_agent:
                response = agent.run(input_text)
            elif use_chat:
                response = chat([HumanMessage(content=input_text),]).content
            else:
                response = llm(input_text)

            print(simple_colorizer(f"... TOTAL TOKENS USED: {cb.total_tokens} ..."))
            print_colored_output(
                input_text, response_text=response.strip(),
                full_color=use_color, input_color=input_clr, response_color=response_clr
            )


def main():
    parser = argparse.ArgumentParser(
        description="Interact with an LLM to demonstrate the streaming functionality within LangChain")
    parser.add_argument("-n", "--model_name", type=str, default="text-davinci-002", help="Model name for the LLM")
    parser.add_argument("-t", "--temperature", type=float, default=0.0, help="Temperature for randomness in output")
    parser.add_argument("-c", "--use_color", action="store_false", help="Color only to titles")
    parser.add_argument("--input_clr", type=str, default="blue", help="Color for user input text")
    parser.add_argument("--response_clr", type=str, default="red", help="Color for AI response text")
    parser.add_argument("-v", "--verbose", action="store_true", help="Flag to set the verbosity of the output.")
    parser.add_argument("-s", "--streaming", action="store_true", help="Flag to add streaming functionality.")
    parser.add_argument("-a", "--use_agent", action="store_true", help="Flag to add agent usage")

    args = parser.parse_args()
    print(args.__dict__)
    # Load the API keys from the .env file
    retrieve_local_api_keys()

    # Call the main function
    llm_interaction(**args.__dict__)


if __name__ == "__main__":
    main()
