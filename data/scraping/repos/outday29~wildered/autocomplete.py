import inspect
from pathlib import Path
from typing import Any, Dict, List, Literal

import pyperclip
import guidance

from wildered.context.utils import temporary_workspace
from wildered.utils import read_file, write_file
from wildered.logger import logger

from .tasks import (
    TaskGroup,
)

# Use OPENAI_API_KEY env
guidance.llm = guidance.llms.OpenAI("text-davinci-003")

def task_executor(
    task_list: List[TaskGroup],
    clipboard: bool = False,
    auto_integrate: bool = False,
) -> None:
    for group in task_list:
        final_prompt = format_task_prompt(group=group, clipboard=clipboard)
        
        if auto_integrate:
            final_prompt = augment_guidance_prompt(prompt=final_prompt)
            response = get_llm_response(final_prompt)
            logger.debug(f"LLM response: {response=}")
            group.integrate(response=response)

def format_task_prompt(group: TaskGroup, clipboard: bool) -> str:
    prompt = group.format_prompt()

    if clipboard:
        print("Copied prompt to clipboard")
        pyperclip.copy(prompt)

    with temporary_workspace() as f:
        write_file(f, prompt)
        print(f"Prompt wrote into {f}")
        _ = input("Press enter to continue/exit. ")
        return read_file(f)


def augment_guidance_prompt(prompt: str) -> str:
    logger.debug(f"Before guidance: {prompt=}")
    additions = inspect.cleandoc("""
        Your answer should only consist of code. All explanation should be done with comments instead of raw text.
        ```python
        {{gen 'code'}}
        ```
    """)
    prompt = prompt + additions
    logger.debug(f"After guidance: {prompt=}")
    return prompt

def get_llm_response(prompt: str) -> str:
    guidance_program = guidance(prompt)
    raw_response = guidance_program()
    cleaned = raw_response['code'].strip("```")
    return cleaned