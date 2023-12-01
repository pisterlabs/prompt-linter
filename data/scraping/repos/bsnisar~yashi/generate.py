# pylint: disable=invalid-name
# pylint: disable=trailing-whitespace

import os
import textwrap

from cohere import AsyncClient


async def generate(prompt: str, api_key: str = os.environ.get("YASHI_COHERE_KEY")) -> str:
    """
    Generate a single-line terminal command based on the provided prompt.

    Args:
        prompt (str): The prompt specifying the desired terminal command.
        api_key (str, optional): The API key for Cohere. Defaults to the value 
        of the "YASHI_COHERE_KEY" environment variable.

    Returns:
        str: The generated terminal command.

    Raises:
        ValueError: If the API key is not provided or set in the environment.

    Example:
        >>> generate_terminal_command("List all files in the current directory")
        'ls'
    """
    if api_key is None:
        raise ValueError('env varialbe "YASHI_COHERE_KEY" is not set')
  
    async with AsyncClient(api_key) as co:  
        response = await co.generate(  
            model='command-nightly', 
            prompt = _generate_prompt(prompt),
            max_tokens=250,  
            temperature=0.750
        )
    return response.generations[0].text
  
def _generate_prompt(prompt: str) -> str:

    # https://docs.cohere.com/docs/prompt-engineering
    task = f'''
        Create a single-line command that can be directly run in the terminal. Do not include any other text.
        The command should do what is specified in the prompt. 
        The prompt is: {prompt}
        Reply with the single line command:
    '''.format(prompt)
    res = textwrap.dedent(task)
    return res
