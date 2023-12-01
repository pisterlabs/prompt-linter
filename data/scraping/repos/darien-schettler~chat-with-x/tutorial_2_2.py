import argparse
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from misc.utils import retrieve_local_api_keys, print_colored_output


def chat_completion_script(model_name="gpt-3.5-turbo", temperature=0,
                           in_lang="English", out_lang="French",
                           input_clr="blue", response_clr="green", system_clr="magenta", use_color=True):
    """ Generate a chat completion using LangChain ChatOpenAI

    Args:
        model_name (str, optional)): Model name for the LLM (default: gpt-3.5-turbo) to be used in the ChatOpenAI instance
        temperature (float, optional)): Temperature for randomness in output. Higher values will result in more random output
        in_lang (str, optional): Language 1 - source languages (default is English)
        out_lang (str, optional): Language 2 - destination languages (default is French)
        input_clr (str, optional)): Color for user input text
        response_clr (str, optional)): Color for AI response text
        system_clr (str, optional)): Color for system message text
        use_color (bool, optional)): Color only to titles

    Returns:
        None; prints the chat completion to the console and interacts with the user
    """

    system_message_template = "You are a helpful assistant that translates {in_lang} to {out_lang}."
    human_message_template = "{text}"

    # Load the API keys from the .env file
    retrieve_local_api_keys()

    # Initialize the ChatOpenAI instance with the specified parameters
    chat = ChatOpenAI(model_name=model_name, temperature=temperature)

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_message_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_message_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    # Continue the conversation loop until the user decides to exit
    while True:

        # Get the user input and break if the user decides to exit
        input_text = input("\nEnter your message (or press enter to exit):\n")
        if not input_text: break

        # Generate the chat completion
        messages = chat_prompt.format_prompt(in_lang=in_lang, out_lang=out_lang, text=input_text).to_messages()
        response = chat(messages)

        print_colored_output(input_text, response.content,
                             system_message_template.format(in_lang=in_lang, out_lang=out_lang),
                             input_clr, response_clr, system_clr,
                             full_color=use_color)


# Example usage:
# python3 -m langchain_quickstart_tutorials.tutorial_2_2 -l2="Inuktitut"
# 'The wide river runs from the North to the South and freezes over in the winter'
def main():
    parser = argparse.ArgumentParser(description="Generate a chat completion using LangChain ChatOpenAI")
    parser.add_argument("-n", "--model_name", type=str, default="gpt-3.5-turbo", help="Model name for the ChatLLM")
    parser.add_argument("-t", "--temperature", type=float, default=0, help="Temperature for randomness in output")
    parser.add_argument("-c", "--use_color", action="store_false", help="Color only to titles")
    parser.add_argument("-l1", "--in_lang", type=str, default="English",
                        help="Input languages for the chat completion (user)")
    parser.add_argument("-l2", "--out_lang", type=str, default="French",
                        help="Output languages for the chat completion (bot)")
    parser.add_argument("--system_clr", type=str, default="magenta", help="Color for system message text")
    parser.add_argument("--input_clr", type=str, default="blue", help="Color for user input text")
    parser.add_argument("--response_clr", type=str, default="green", help="Color for AI response text")
    args = parser.parse_args()

    # Generate the chat completion based on the user inputs
    chat_completion_script(**args.__dict__)


if __name__ == "__main__":
    main()
