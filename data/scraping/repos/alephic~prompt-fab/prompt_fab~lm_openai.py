from typing import Optional
import openai
import os
from prompt_fab.templates import Template

__all__ = [
    "get_completion_tokens_and_logprobs",
    "get_template_completion_tokens_and_logprobs"
]

if 'OPENAI_API_KEY' not in os.environ and openai.api_key_path is None:
    openai.api_key_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),  'openai_api_key.txt')
    assert os.path.exists(openai.api_key_path)

DEFAULT_MODEL = 'text-davinci-002'

def get_completion_tokens_and_logprobs(prompt_text: str, completion_text: str, model: Optional[str]=None):
    """
    Accepts a prompt prefix and a desired completion and returns a tuple of the
    tokens and conditional token log-likelihoods corresponding to the provided completion.
    """
    if model is None:
        model = DEFAULT_MODEL
    response = openai.Completion.create(
        model=model,
        prompt=prompt_text+completion_text,
        max_tokens=0,
        logprobs=0,
        echo=True,
        n=1
    )
    logprobs_obj = response["choices"][0]["logprobs"]
    completion_start_index = None
    for i, offset in enumerate(logprobs_obj["text_offset"]):
        if offset == len(prompt_text):
            completion_start_index = i
            break
        elif offset > len(prompt_text):
            completion_start_index = i-1
            break
    if completion_start_index is None:
        completion_start_index = len(logprobs_obj['text_offset'])-1
    return logprobs_obj["tokens"][completion_start_index:], logprobs_obj["token_logprobs"][completion_start_index:]

def get_template_completion_tokens_and_logprobs(template: Template, data_partial, data_full, model: Optional[str]=None):
    """
    Accepts a prompt template and two versions of a data object:
        data_partial: The data specifying just the prompt prefix
        data_full: The full data, which also specifies the target completion
    Returns a tuple containing the tokens and conditional token log-likelihoods corresponding
    to only the portion of the filled template that is not specified by data_partial.
    """
    prompt_text = template.fill(data_partial)
    full_text = template.fill(data_full)
    assert full_text.startswith(prompt_text)
    completion_text = full_text[len(prompt_text):]
    return get_completion_tokens_and_logprobs(prompt_text, completion_text, model=model)