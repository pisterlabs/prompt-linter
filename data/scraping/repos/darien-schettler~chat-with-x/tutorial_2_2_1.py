import ast
import argparse

from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.chat_models import ChatOpenAI
from misc.utils import print_colored_output, retrieve_local_api_keys, get_streaming_cb, simple_colorizer
from langchain import PromptTemplate, HuggingFaceHub, LLMChain


def llm_few_shot(system_message, examples, model_name="gpt-3.5-turbo", temperature=0.7,  hf_repo_id="", max_length=2048,
                 use_color=True, input_clr="blue", response_clr="red", verbose=False, **kwargs):
    """ Interact with an LLM to demonstrate few shot reasoning

    Args:
        system_message (str): The system message we will use.
        examples (dict): A dictionary of examples to use for the few shot reasoning
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

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_message)

    examples_humans, example_ais = [], []
    for ex_human, ex_ai in examples.items():
        examples_humans.append(SystemMessagePromptTemplate.from_template(
            ex_human, additional_kwargs={"name": "example_user"}
        ))
        example_ais.append(SystemMessagePromptTemplate.from_template(
            ex_ai, additional_kwargs={"name": "example_assistant"}
        ))

    # interleave the examples_human messages and example ai messages
    ex_templates = [item for t in zip(examples_humans, example_ais) for item in t]

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    # Create the chat prompt as if there was an existing conversation between the user and the LLM
    # that showcases the usage that is expected.
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, *ex_templates, human_message_prompt]
    )

    # If HF
    if hf_repo_id != "":
        llm_chain = LLMChain(
            prompt=chat_prompt, llm=HuggingFaceHub(
                repo_id=hf_repo_id, model_kwargs={"max_length": max_length}
            ), verbose=verbose,
        )
    else:
        # get a chat completion from the formatted messages
        chat = ChatOpenAI(
            model_name=model_name, temperature=temperature, streaming=True,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=verbose
        )
        llm_chain = LLMChain(prompt=chat_prompt, llm=chat, verbose=verbose)

    while True:
        # Get the user input and break if the user decides to exit
        input_text = input("\nEnter your message (or press enter to exit):\n")
        if not input_text: break

        response = llm_chain.run(input_text)
        print_colored_output(
            input_text, response_text=response.strip(),
            full_color=use_color, input_color=input_clr, response_color=response_clr
        )


def main():
    parser = argparse.ArgumentParser(
        description="Interact with an LLM after providing a few examples of the input and output."
    )

    parser.add_argument("--system_message", type=str,
                        default="You are a helpful assistant that provides examples of how to use a word that the user"
                                "defines in a sentence. The user will provide a word and you will provide an example of"
                                "how to use that word in a sentence. The word may be made up and that's ok.",
                        help="The system prompt we will use.")

    parser.add_argument("--examples", type=str, nargs="+",
                        default='{"A -whatpu- is a small, furry animal native to Tanzania.": "An example of a '
                                'sentence that uses the word whatpu is: We were traveling in Africa and we saw these '
                                'very cute whatpus.",}',
                        help="The example(s) we will use to show the LLM how to behave")

    # Chat model arguments - TBD --> gpt-4
    parser.add_argument("-n", "--model_name", type=str, default="gpt-3.5-turbo", help="Model name for the LLM")
    parser.add_argument("-t", "--temperature", type=float, default=0.7, help="Temperature for randomness in output")

    # Alternative model arguments
    parser.add_argument("-hf", "--hf_repo_id", type=str, default="",  # google/flan-t5-xl
                        help="The HuggingFace repo ID for the model")
    parser.add_argument("--max_length", type=int, default=2048, help="The model specific max length")

    # Colour for prints
    parser.add_argument("-c", "--use_color", action="store_false", help="Color only to titles")
    parser.add_argument("--input_clr", type=str, default="blue", help="Color for user input text")
    parser.add_argument("--response_clr", type=str, default="red", help="Color for AI response text")
    parser.add_argument("-v", "--verbose", action="store_true", help="Flag to set the verbosity of the output.")

    args = parser.parse_args()
    args.examples = ast.literal_eval(args.examples)

    # Load the API keys from the .env file
    retrieve_local_api_keys()

    # Call the main function
    llm_few_shot(**args.__dict__)


if __name__ == "__main__":
    main()
