"""
This file has helper functions and abstractions over LLMs used for
model-generated evals (creating or scoring questions). Importantly, what
actually RUNS the evals is different, and specified by wrappers and configs for
OpenAI Evals in the evals_completers folder.
"""

import time
import openai
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import replicate
from pydantic import BaseModel as BM
from typing import Any, List, Optional, Sequence, Tuple, Dict, Union
import fnmatch
from evalugator.utils import flatten, render_jinja_string
from evalugator.structs import Message
from evalugator.local_models.llama2 import llama_text as llama_text_local


def openai_chat_text_fn(model, **kwargs):
    def f(prompt, **new_kwargs):
        result = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **{**kwargs, **new_kwargs},
        )
        result_choices = result.choices  # type: ignore
        if len(result_choices) == 1:
            return result_choices[0]["message"]["content"]
        else:
            return [choice["message"]["content"] for choice in result_choices]

    return f


def openai_text_fn(model, **kwargs):
    def f(prompt, **new_kwargs):
        result = openai.Completion.create(
            model=model, prompt=prompt, **{**kwargs, **new_kwargs}
        )
        result_choices = result.choices  # type: ignore
        if len(result_choices) == 1:
            return result_choices[0]["text"]
        else:
            return [choice["text"] for choice in result_choices]

    return f


def openai_chat(model, messages, **kwargs):
    result = openai.ChatCompletion.create(model=model, messages=messages, **kwargs)
    result_choices = result.choices  # type: ignore
    if len(result_choices) == 1:
        return result_choices[0]["message"]["content"]
    else:
        return [choice["message"]["content"] for choice in result_choices]


def openai_text(model, prompt, **kwargs):
    result = openai.Completion.create(model=model, prompt=prompt, **kwargs)
    result_choices = result.choices  # type: ignore
    if len(result_choices) == 1:
        return result_choices[0]["text"]
    else:
        return [choice["text"] for choice in result_choices]


def anthropic_chat(model, prompt, **kwargs):
    anthropic = Anthropic()
    completion = anthropic.completions.create(model=model, prompt=prompt, **kwargs)
    return completion.completion


replicate_model_version_lookup = {
    "llama-2-7b": "meta/llama-2-7b:527827021d8756c7ab79fde0abbfaac885c37a3ed5fe23c7465093f0878d55ef",
    "llama-2-7b-chat": "meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e",
    "llama-2-13b": "meta/llama-2-13b:078d7a002387bd96d93b0302a4c03b3f15824b63104034bfa943c63a8f208c38",
    "llama-2-13b-chat": "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d",
    "llama-2-70b": "meta/llama-2-70b:a52e56fee2269a78c9279800ec88898cecb6c8f1df22a6483132bea266648f00",
    "llama-2-70b-chat": "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
}


def replicate_text(
    model, prompt, **kwargs
):  # max_new_tokens=128, min_new_tokens=-1, temperature=0.75, stop_sequence):
    input = {"prompt": prompt}
    for k, v in kwargs.items():
        input[k] = v
    model_id = replicate_model_version_lookup[model]
    out = replicate.run(model_id, input=input)
    s = ""
    for i in out:
        s += i
    return s


def str2message(s: str):
    return Message(content=s)


def str2chat(s: str):
    return [Message(content=s, role="user")]


def messages_to_text(messages: List[Message]):
    return "\n\n".join([f"{m.content}" for m in messages])


def messages_to_llama_messages(messages: List[Message]):
    llama_format = """[INST] {{user_message}} [/INST]"""
    llama_format_assistant = """{{assistant_message}}"""
    llama_format_system = """<s>[INST] <<SYS>>
{{system_prompt}}
<</SYS>>

{{user_message}} [/INST]
"""
    llama_text = ""
    i = 0
    if messages[0].role == "system":
        llama_text += render_jinja_string(
            llama_format_system,
            system_prompt=messages[0].content,
            user_message=messages[1].content,
        )
        i = 2
    while i < len(messages):
        m = messages[i]
        if m.role == "user":
            llama_text += render_jinja_string(llama_format, user_message=m.content)
        elif m.role == "assistant":
            llama_text += render_jinja_string(
                llama_format_assistant, assistant_message=m.content
            )
        elif m.role == "system":
            raise ValueError(
                "System messages for Llama can only be the first message in a list of messages"
            )
    return llama_text


model_categories = {
    "gpt-4": "openai_chat",
    "gpt-4-[!b]*": "openai_chat",
    "gpt-4-base": "openai_text",
    "gpt-3.5-turbo*": "openai_chat",
    "text-davinci-*": "openai_text",
    "text-curie-*": "openai_text",
    "text-babbage-*": "openai_text",
    "text-ada-*": "openai_text",
    "code-davinci-*": "openai_text",
    "ada*": "openai_text",
    "babbage*": "openai_text",
    "curie*": "openai_text",
    "davinci*": "openai_text",
    "claude-2": "anthropic_chat",
    "llama-2-*-chat": "llama_chat",
    "llama-2-7b": "replicate_text",
    "llama-2-13b": "replicate_text",
    "llama-2-70b": "replicate_text",
    "dummy": "dummy_chat",
}


def get_model_category(model_name):
    for pattern, category in model_categories.items():
        if fnmatch.fnmatch(model_name, pattern):
            return category
    raise ValueError(f"Could not find category for model {model_name}")


model_input_converters = {
    "openai_chat": lambda msgs: [msg.model_dump() for msg in msgs],
    "openai_text": messages_to_text,
    "anthropic_chat": lambda msgs: f"{HUMAN_PROMPT} {messages_to_text(msgs)}{AI_PROMPT}"[
        1:
    ],  # remove initial \n
    "llama_chat": messages_to_llama_messages,
    "replicate_text": messages_to_text,
    "dummy_chat": lambda msgs: [msg.content for msg in msgs],
}

model_fns = {
    "openai_chat": openai_chat,
    "openai_text": openai_text,
    "anthropic_chat": anthropic_chat,
    "llama_chat": llama_text_local,
    "replicate_text": replicate_text,
    "dummy_chat": lambda model, messages, **kwargs: "dummy completion",
}


def model_input_converter(model: str, messages: List[Message]):
    category = get_model_category(model)
    assert (
        category in model_input_converters.keys()
    ), f"Unknown model category {category} for model {model}"
    converter = model_input_converters[category]
    return converter(messages)


class KwargNameOptions(BM):
    openai: Optional[str] = None
    anthropic: Optional[str] = None
    replicate: Optional[str] = None
    llama: Optional[str] = None


kwarg_name_options = [
    KwargNameOptions(
        openai="temperature", anthropic="temperature", replicate="temperature"
    ),
    KwargNameOptions(
        openai="max_tokens",
        anthropic="max_tokens_to_sample",
        replicate="max_new_tokens",
        llama="max_new_tokens",
    ),
    KwargNameOptions(
        openai="stop", anthropic="stop_sequences", replicate="stop_sequences"
    ),
]


def kwarg_converter(model: str, kwargs: Dict[str, Any]):
    companies = list(
        set(
            flatten(
                [
                    list(vars(kwarg_name_option).keys())
                    for kwarg_name_option in kwarg_name_options
                ]
            )
        )
    )
    kwargs_by_company = {
        company: [
            getattr(kwarg_name_option, company)
            for kwarg_name_option in kwarg_name_options
            if getattr(kwarg_name_option, company) is not None
        ]
        for company in companies
    }
    all_kwargs = flatten(kwargs_by_company.values())
    new_kwargs = {}
    to_company = get_model_category(model).split("_")[0]
    for kwarg_name, val in kwargs.items():
        if (
            to_company in kwargs_by_company.keys()
            and kwarg_name in kwargs_by_company[to_company]
        ) or kwarg_name not in all_kwargs:
            new_kwargs[kwarg_name] = val
        else:
            for company in companies:
                if company in kwargs_by_company.keys():
                    if kwarg_name in kwargs_by_company[company]:
                        kwarg_name_option = list(
                            filter(
                                lambda kno: getattr(kno, company) == kwarg_name,
                                kwarg_name_options,
                            )
                        )[0]
                        new_name = getattr(kwarg_name_option, to_company)
                        new_kwargs[new_name] = val
                        break
    return new_kwargs


def model_output(
    model: str, messages: List[Message], convert_kwargs=True, retry=False, **kwargs
):
    """
    Returns output from a model called `model`, assumed to be a key in the
    `model_categories` dictionary in llms.py. Messages is a list of `Message`,
    which is a an abstraction of role & content -containing objects that are
    auto-converted to either plain text, Anthropic-style messages, or OpenAI's
    expected chat format (use `str2chat` on a string if you want to completely
    ignore this).

    If `convert_kwargs` is passed as True (as it is by default), the function
    will do some auto-conversion where some model arguments like temperature /
    max tokens / etc. will be auto-converted between the OpenAI and Anthropic
    kwarg-naming conventions.
    """
    if convert_kwargs:
        kwargs = kwarg_converter(model, kwargs)
    input = model_input_converter(model, messages)
    category = get_model_category(model)
    if retry:
        # TODO: replace with proper retry library
        waiting = 1
        out = None
        while True:
            try:
                out = model_fns[category](model, input, **kwargs)
                if out is not None:
                    break
            except:
                print(f"Retrying {model} output generation in {waiting} seconds...")
                time.sleep(waiting)
                waiting *= 2
    else:
        out = model_fns[category](model, input, **kwargs)
    return out


def model_output_from_str(model: str, prompt: str, **kwargs):
    return model_output(model, [str2message(prompt)], **kwargs)
