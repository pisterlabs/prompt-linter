import argparse
from misc.utils import print_colored_output, retrieve_local_api_keys, get_streaming_cb, simple_colorizer
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, HuggingFaceHub
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


def llm_chat(system_prompt_template=None, human_prompt_template=None, in_lang="English", out_lang="French",
             model_name="gpt-3.5-turbo", temperature=0.7,  hf_repo_id="", max_length=2048,
             use_color=True, input_clr="blue", response_clr="red", verbose=False, **kwargs):
    """ Interact with a HuggingFace Hub LLM to demonstrate the LangChain integration

    Args:
        system_prompt_template (str): The system message template we will use. Default is translation.
        human_prompt_template (str): The human message template we will use. Default is a pass through
        in_lang (str, optional): Language 1 - source languages (default is English)
        out_lang (str, optional): Language 2 - destination languages (default is French)
        model_name (str): The model name for the LLM
        temperature (float): The temperature for the LLM
        hf_repo_id (str): The HuggingFace repo ID for the model
        max_length (int): The maximum length of the output
        use_color (bool, optional): Flag to use color in the output
        input_clr (str, optional): Color for user input text
        response_clr (str, optional): Color for AI response text
        verbose (bool, optional): Flag to set the verbosity of the output.
        **kwargs: Additional keyword arguments

    Returns:
        None; interacts with the user and prints the user inputs and model responses to the console
    """

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_prompt_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    # If HF
    if hf_repo_id != "":
        llm_chain = LLMChain(
            prompt=chat_prompt, llm=HuggingFaceHub(
                repo_id=hf_repo_id, model_kwargs={"max_length": max_length}
            ), verbose=verbose
        )
    else:
        # get a chat completion from the formatted messages
        chat = ChatOpenAI(model_name=model_name, temperature=temperature)
        llm_chain = LLMChain(prompt=chat_prompt, llm=chat, verbose=verbose)

    while True:
        # Get the user input and break if the user decides to exit
        input_text = input("\nEnter your message (or press enter to exit):\n")
        if not input_text: break

        response = llm_chain.run(input_language=in_lang, output_language=out_lang, text=input_text)
        print_colored_output(
            input_text, response_text=response.strip(),
            full_color=use_color, input_color=input_clr, response_color=response_clr
        )


def main():
    parser = argparse.ArgumentParser(
        description="Interact with a HuggingFace Hub LLM to demonstrate the LangChain integration")

    # Prompt arguments
    parser.add_argument("--system_prompt_template", type=str,
                        default="You are a helpful assistant that translates {input_language} to {output_language}.",
                        help="The system prompt template we will use. Default is translation.")
    parser.add_argument("--human_prompt_template", type=str,
                        default="{text}",
                        help="The human prompt template we will use. Default is passthrough for input.")
    parser.add_argument("-l1", "--in_lang", type=str, default="English",
                        help="Input languages for the chat completion (user)")
    parser.add_argument("-l2", "--out_lang", type=str, default="French",
                        help="Output languages for the chat completion (bot)")

    # Chat model arguments - TBD --> gpt-4
    parser.add_argument("-n", "--model_name", type=str, default="gpt-3.5-turbo", help="Model name for the LLM")
    parser.add_argument("-t", "--temperature", type=float, default=0.7, help="Temperature for randomness in output")

    # Alternative model arguments
    parser.add_argument("-hf", "--hf_repo_id", type=str, default="", # google/flan-t5-xl
                        help="The HuggingFace repo ID for the model")
    parser.add_argument("--max_length", type=int, default=2048, help="The model specific max length")

    # Colour for prints
    parser.add_argument("-c", "--use_color", action="store_false", help="Color only to titles")
    parser.add_argument("--input_clr", type=str, default="blue", help="Color for user input text")
    parser.add_argument("--response_clr", type=str, default="red", help="Color for AI response text")
    parser.add_argument("-v", "--verbose", action="store_true", help="Flag to set the verbosity of the output.")

    args = parser.parse_args()
    print(args.__dict__)

    # Load the API keys from the .env file
    retrieve_local_api_keys()

    # Call the main function
    llm_chat(**args.__dict__)


if __name__ == "__main__":
    main()
