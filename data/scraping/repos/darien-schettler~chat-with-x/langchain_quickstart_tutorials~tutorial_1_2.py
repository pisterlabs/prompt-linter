import argparse
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from colorama import init, Fore

from misc.utils import retrieve_local_api_keys


def generate_x_from_input(input_obj,
                          prompt_template_prefix="What is a good name for a company that makes",
                          model_name='text-davinci-003',
                          temperature=0.7, **kwargs):
    """ Generate something based on the users input. Defaults to company name.

    This script takes a user input that will be used to generate something specific based on some
    predefined (or passed by the user) prompt template. The default template is for generating a company name
    It uses the `PromptTemplate` class from `langchain.prompts` to define and format the prompt, then sends
    it to the specified LLM for generating a new company name.

    Args:
        input_obj (str): user input describing whatever the user wants to generate (defaults to company name)
        prompt_template_prefix (str, optional): Prefix for the prompt template. Defaults to company name related prompt.
        model_name (str, optional): Model name for the LLM
        temperature (float, optional): Temperature for randomness in output – Lower values are more random.

    Returns:
        None; prints the generated company name
    """
    # Load the API keys from the .env file
    retrieve_local_api_keys()

    # Define the prompt template prior to user input injection (obj to be replaced with `obj_input`)
    prompt_template = prompt_template_prefix + " {obj}?"

    # Define the prompt template
    prompt = PromptTemplate(
        input_variables=["obj"],
        template=prompt_template,
    )

    # Initialize the LLM wrapper with the specified parameters
    llm = OpenAI(model_name=model_name, temperature=temperature)

    # Create the LLMChain
    chain = LLMChain(llm=llm, prompt=prompt)

    # Run the LLMChain with the input object
    response = chain.run(input_obj)

    # Print the formatted prompt and response
    formatted_prompt = prompt.format(obj=input_obj)
    print(f"\n{Fore.BLUE}Input Prompt   :{Fore.RESET} '{formatted_prompt}'")
    print(f"{Fore.GREEN}Response       :{Fore.RESET} '{response.strip()}'\n")


def main():
    """ Execute the main script to generate a company name based on the input product """
    parser = argparse.ArgumentParser(description="Generate a company name using LangChain LLM")
    parser.add_argument("input_obj", type=str, help="Description of the object the user wants to generate around")
    parser.add_argument("-ptp", "--prompt_template_prefix", type=str,
                        default="What is a good name for a company that makes",
                        help="Prefix for the prompt template. The user input will be appended to this.")
    parser.add_argument("-n", "--model_name", type=str, default="text-davinci-003", help="Model name for the LLM")
    parser.add_argument("-t", "--temperature", type=float, default=0.7, help="Temperature for randomness in output")
    args = parser.parse_args()

    # Generate the company name based on the user inputs
    generate_x_from_input(**args.__dict__)


# Example usage:
# python3 -m langchain_quickstart_tutorials.tutorial_1_2 \
# "inner elbow crease" \
# -ptp="If you had only ever read shakespeare and plato and your best friend was a pineapple \
# and a talking cat, what would you name your"  -t=1.75
if __name__ == "__main__":
    main()
