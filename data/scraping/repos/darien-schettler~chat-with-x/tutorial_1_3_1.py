import argparse
from misc.utils import print_colored_output, retrieve_local_api_keys, get_streaming_cb, simple_colorizer
from langchain import PromptTemplate, HuggingFaceHub, LLMChain


def llm_integration(prompt_template, hf_repo_id, use_color=True, input_clr="blue", response_clr="red", verbose=False,
                    **kwargs):
    """ Interact with a HuggingFace Hub LLM to demonstrate the LangChain integration

    Args:
        hf_repo_id (str): The HuggingFace repo ID for the model
        prompt_template (str): The prompt template we will use.
        use_color (bool, optional): Flag to use color in the output
        input_clr (str, optional): Color for user input text
        response_clr (str, optional): Color for AI response text
        verbose (bool, optional): Flag to set the verbosity of the output.
        **kwargs: Additional keyword arguments

    Returns:
        None; interacts with the user and prints the user inputs and model responses to the console
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=HuggingFaceHub(repo_id=hf_repo_id, model_kwargs={"max_length":100}),
                         verbose=verbose)

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
        description="Interact with a HuggingFace Hub LLM to demonstrate the LangChain integration")
    parser.add_argument("--prompt_template", type=str,
                        default="Question: {question}\n\nAnswer: Let's think step by step.",
                        help="The prompt template we will use.")
    parser.add_argument("-hf", "--hf_repo_id", type=str, default="google/flan-t5-xl",
                        help="The HuggingFace repo ID for the model")
    parser.add_argument("-t", "--temperature", type=float, default=0.7, help="Temperature for randomness in output")
    parser.add_argument("--max_length", type=int, default=2048, help="The model specific max length")
    parser.add_argument("-c", "--use_color", action="store_false", help="Color only to titles")
    parser.add_argument("--input_clr", type=str, default="blue", help="Color for user input text")
    parser.add_argument("--response_clr", type=str, default="red", help="Color for AI response text")
    parser.add_argument("-v", "--verbose", action="store_true", help="Flag to set the verbosity of the output.")

    args = parser.parse_args()
    print(args.__dict__)

    # Load the API keys from the .env file
    retrieve_local_api_keys()

    # Call the main function
    llm_integration(**args.__dict__)


if __name__ == "__main__":
    main()