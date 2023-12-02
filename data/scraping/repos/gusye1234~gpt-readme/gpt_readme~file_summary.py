# import sys
import openai
from .constants import console, envs
from .utils import (
    get_file_content,
    construct_prompt,
    get_language,
    hash_content,
    relative_module,
)
from .prompts import FILE_PROMPT, SYSTEM_PROMPT


async def prompt_summary(**kwargs):
    content = kwargs['code']
    if envs["cache"] is not None:
        file_cache = envs["cache"].get(kwargs['path'], None)
        if file_cache is not None and file_cache["hash"] == hash_content(content):
            return file_cache["summary"]
    final_prompt = FILE_PROMPT.format(**kwargs)
    final_system = SYSTEM_PROMPT.format(**kwargs)
    response = await openai.ChatCompletion.acreate(
        model=envs['gpt_model'],
        messages=construct_prompt(final_system, final_prompt),
        temperature=0,
    )
    output = response["choices"][0]["message"]["content"]
    if envs["cache"] is not None:
        envs["cache"][kwargs['path']] = {}
        envs["cache"][kwargs['path']]["summary"] = output
        envs["cache"][kwargs['path']]["hash"] = hash_content(content)
    return output


async def file_summary(file_path):
    module = relative_module(file_path)
    console.print(f"[bold blue]FILE[/bold blue] {module} running...")
    content = "".join(get_file_content(file_path)).strip()
    language = get_language(file_path)
    summary = await prompt_summary(
        language=language, code=content, max_length=200, path=module
    )
    console.print(f"[bold green]✓ FILE[/bold green] {module} done")
    return {"summary": summary, "language": language}
