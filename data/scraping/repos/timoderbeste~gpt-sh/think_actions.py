import os

from openai_client import get_gpt_response
from prompt_builder import PromptBuilder
from utils import typer_writer

prompt_builder = PromptBuilder()


def handle_think_action(inp, env_var2val, temperature):
    inp = inp.replace("THINK: ", "")

    for var in env_var2val:
        inp = inp.replace(var, (env_var2val[var] if env_var2val[var] else ""))

    prompt = inp
    # print("Your THINK prompt is: ", prompt)
    response = get_gpt_response(
        prompt, temperature=temperature, top_p=1, caching=False, chat=None)
    typer_writer(response)
    return response
