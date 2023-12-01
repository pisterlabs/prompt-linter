#!/usr/bin/env python

import contextlib
import os
import re
import time
from datetime import datetime
from typing import Any, Dict

import click
import yaml
from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import LLMChain, SequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from md2pdf.core import md2pdf

PROMPT_TEMPLATES_DIR = "templates"
COMMON_PREFIX_FILE = "_common.txt"
CONFIG_FILE = "config.yaml"
OUTPUT_TEMPLATE_FILE = "output.txt"
EXAMPLE_INPUT_FILE = "input_example.md"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MODEL_NAME = "gpt-3.5-turbo-16k"


def extract_variable_names(template):
    """
    Extract and return variable names from a template string.

    Parameters:
    - template (str): The template string containing variables enclosed in curly braces {}.

    Returns:
    - list: A list of variable names extracted from the template string.
    """
    return re.findall(r"{(.*?)}", template)


def read_prompt_template(template_name, prompt_templates_dir, common_prefix_file):
    """
    Read and return content from a prompt template file with a common prefix added to it.

    Parameters:
    - template_name (str): Name of the template file.
    - prompt_templates_dir (str): Directory path containing prompt template files.
    - common_prefix_file (str): Name of the file containing common prefix content.

    Returns:
    - str: Content of the template file with common prefix added.
    """
    common_prefix_path = os.path.join(prompt_templates_dir, common_prefix_file)
    with open(common_prefix_path, "r") as common_file:
        common_prefix = common_file.read()

    template_path = os.path.join(prompt_templates_dir, template_name)
    with open(template_path, "r") as template_file:
        return common_prefix + template_file.read()


def load_chain_config(config_file):
    """
    Load and return configuration from a YAML file.

    Parameters:
    - config_file (str): The path to the configuration YAML file.

    Returns:
    - dict: Configuration data loaded from the file.
    """
    try:
        with open(config_file, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        return {}


def read_template(template_name):
    """
    Read and return content from a template file.

    Parameters:
    - template_name (str): Name of the template file.

    Returns:
    - str: Content of the template file.
    """
    path = os.path.join(PROMPT_TEMPLATES_DIR, template_name)
    with open(path, "r") as f:
        return f.read()


def create_llm_chain(llm, template_file, prompt_templates_dir, common_prefix_file):
    """
    Create and return an LLMChain instance configured with the given parameters.

    Parameters:
    - llm (ChatOpenAI): An instance of ChatOpenAI which is used to perform the language model operations.
    - template_file (str): The name of the template file to be used for prompt creation.
    - prompt_templates_dir (str): Directory path containing prompt template files.
    - common_prefix_file (str): Name of the file containing common prefix content to be appended before the template.

    Returns:
    - LLMChain: An instance of LLMChain configured with the given parameters.
    """
    monitor = CallbackHandler()
    # Extract variable names as input_keys
    template_content = read_prompt_template(
        template_file, prompt_templates_dir, common_prefix_file
    )
    input_keys = extract_variable_names(template_content)
    # Set output_key as the name of the template file without the file extension
    output_key = os.path.splitext(template_file)[0]
    return LLMChain(
        llm=llm,
        prompt=PromptTemplate(input_variables=input_keys, template=template_content),
        output_key=output_key,
        callbacks=[monitor],
        tags=[output_key],
    )


def build_chain(
    api_key,
    chains_config,
    prompt_templates_dir,
    common_prefix_file,
    verbose=False,
    model_name="gpt-3.5-turbo-16k",
    temperature=0.7,
):
    """
    Build and return a SequentialChain by running several LLMChains in sequence.

    Parameters:
    - api_key (str): The API key to access the language model.
    - chains_config (list): A list of dictionaries, each containing configuration for a chain (e.g., template file).
    - prompt_templates_dir (str): Directory path containing prompt template files.
    - common_prefix_file (str): Name of the file containing common prefix content to be appended before the template.
    - verbose (bool, optional): If True, prints verbose output. Defaults to False.
    - model_name (str, optional): The name of the language model to be used. Defaults to "gpt-3.5-turbo-16k".
    - temperature (float, optional): The temperature parameter for the language model. Defaults to 0.7.

    Returns:
    - SequentialChain: An instance of SequentialChain configured with the chains created from chains_config.
    """

    # Initialize ChatOpenAI
    llm = ChatOpenAI(openai_api_key=api_key, model=model_name, temperature=temperature)

    # Chains created using the create_llm_chain function
    chains = [
        create_llm_chain(
            llm, chain_config["template_file"], prompt_templates_dir, common_prefix_file
        )
        for chain_config in chains_config
    ]

    # Calculate input_variables and output_variables
    input_variables = extract_variable_names(
        read_prompt_template(
            chains_config[0]["template_file"], prompt_templates_dir, common_prefix_file
        )
    )
    output_variables = [
        os.path.splitext(chain_config["template_file"])[0]
        for chain_config in chains_config
    ]

    # Sequential chain
    sequential_chain = SequentialChain(
        chains=chains,
        input_variables=input_variables,
        output_variables=output_variables,
        verbose=verbose,
    )

    return sequential_chain


class CallbackHandler(BaseCallbackHandler):
    """
    Custom callback handler class for monitoring the progress of the chains.

    This class is a subclass of BaseCallbackHandler and is used to output
    progress information when a chain starts executing.

    Attributes:
        None
    """

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """
        Callback function that is executed when a chain starts.

        Parameters:
        - serialized (dict): The serialized chain information.
        - inputs (dict): The inputs passed to the chain.
        - kwargs (dict): Additional keyword arguments containing tags.

        Returns:
        - None
        """
        click.secho(f"Running chain '{''.join(kwargs['tags'])}'", fg="cyan")


def generate_report(output_file, markdown, **chain_output_dict):
    """
    Generates a report by converting chain output to markdown and then to PDF.

    Parameters:
    - output_file (str): The base name of the output file.
    - markdown (bool): If True, saves the markdown content to a file.
    - chain_output_dict (dict): Dictionary containing the output of the chains.

    Returns:
    - tuple: The names of the created markdown and PDF files.
    """
    output_template = read_template(OUTPUT_TEMPLATE_FILE)
    markdown_output = output_template.format(**chain_output_dict)
    file_name = output_file or f"output-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    markdown_file_name = f"{file_name}.md"
    pdf_file_name = f"{file_name}.pdf"

    # Save markdown content to file
    if markdown:
        with open(markdown_file_name, "w") as f:
            f.write(markdown_output)

    # Convert the markdown file to PDF
    md2pdf(pdf_file_name, md_content=markdown_output)

    # Return the names of the created files
    return markdown_file_name, pdf_file_name


def report_results(markdown, markdown_file_name, pdf_file_name, cb, duration):
    """
    Reports the results of the report generation including file names,
    total tokens, cost, and runtime.

    Parameters:
    - markdown (bool): If True, indicates markdown file was created.
    - markdown_file_name (str): The name of the markdown file.
    - pdf_file_name (str): The name of the PDF file.
    - cb (CallbackHandler): The callback handler used during report generation.
    - duration (float): The total runtime in seconds.

    Returns:
    - None
    """
    if markdown:
        click.secho(f"Markdown file created: {markdown_file_name}", fg="green")
    click.secho(f"PDF file created: {pdf_file_name}", fg="green")
    click.secho(f"Total tokens: {cb.total_tokens}", fg="yellow")
    click.secho(f"Total cost: ${cb.total_cost:.2f}", fg="yellow")
    click.secho(f"Runtime: {duration:.2f} seconds", fg="yellow")


def check_api_key():
    """
    Checks if the OPENAI_API_KEY environment variable is set.

    Returns:
    - str: The API key if it is set.

    Raises:
    - SystemExit: If the OPENAI_API_KEY environment variable is not set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        click.secho("Error: OPENAI_API_KEY environment variable is not set.", fg="red")
        click.secho(
            "Please set it by running: export OPENAI_API_KEY=your_api_key", fg="red"
        )
        exit(1)
    return api_key


def read_seed(seed_file):
    """
    Reads the content of a seed file or displays an example input if no file is provided.

    Parameters:
    - seed_file (str): The name of the seed file.

    Returns:
    - str: The contents of the seed file.
    """
    if seed_file is None:
        click.secho(f"{read_template(EXAMPLE_INPUT_FILE)}", fg="white")
        exit(0)
    else:
        click.secho(f"Using seed file: {seed_file}", fg="green")
        with open(seed_file, "r") as f:
            return f.read()


@contextlib.contextmanager
def measure_time():
    """
    Context manager for measuring the execution time of a code block.

    Yields:
    - function: A function that when called, returns the elapsed time in seconds.
    """
    start_time = time.time()
    yield lambda: time.time() - start_time


@click.command()
@click.option("--seed-file", default=None, help="Path to the seed file.")
@click.option(
    "--output-file", default=None, help="Specify the name of the output file."
)
@click.option(
    "--markdown", is_flag=True, default=False, help="Save output as markdown."
)
@click.option("--verbose", is_flag=True, default=False, help="Enable verbose output.")
@click.option(
    "--config-file", default="config.yaml", help="Path to the configuration file."
)
@click.option("--temperature", default=None, type=float, help="Set the temperature.")
@click.option("--model-name", default=None, type=str, help="Set the model name.")
def main(
    seed_file, output_file, markdown, verbose, config_file, temperature, model_name
):
    """Generate a business model from a hunch file."""

    # Check API Key
    api_key = check_api_key()

    # Read seed file
    seed = read_seed(seed_file)

    # Load the configuration from the specified configuration file
    chain_config = load_chain_config(config_file)

    # Override temperature and model_name if provided
    temperature = temperature or chain_config.get("temperature", DEFAULT_TEMPERATURE)
    model_name = model_name or chain_config.get("model_name", DEFAULT_MODEL_NAME)

    # Get prompt_templates_dir and common_prefix_file from config or set defaults
    prompt_templates_dir = chain_config.get(
        "prompt_templates_dir", PROMPT_TEMPLATES_DIR
    )
    common_prefix_file = chain_config.get("common_prefix_file", COMMON_PREFIX_FILE)

    with measure_time() as duration, get_openai_callback() as cb:
        # Build and execute chain
        chain = build_chain(
            api_key,
            chain_config["chains"],
            prompt_templates_dir,
            common_prefix_file,
            verbose=verbose,
            model_name=model_name,
            temperature=temperature,
        )
        output = chain({"seed": seed})

        # Generate report
        markdown_file_name, pdf_file_name = generate_report(
            output_file, markdown, **output
        )

        # Reporting on result.
        report_results(markdown, markdown_file_name, pdf_file_name, cb, duration())


if __name__ == "__main__":
    load_dotenv()
    main()
