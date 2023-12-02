import os
import openai
import sys
import asyncio
from .utils import (
    ignore_dir,
    ignore_file,
    construct_summary_pair,
    construct_prompt,
    hash_dir,
    relative_module,
)
from .constants import console, envs
from .file_summary import file_summary
from .prompts import MODULE_PROMPT, SYSTEM_PROMPT


async def prompt_summary(**kwargs):
    final_prompt = MODULE_PROMPT.format(**kwargs)
    final_system = SYSTEM_PROMPT.format(**kwargs)
    response = await openai.ChatCompletion.acreate(
        model=envs['gpt_model'],
        messages=construct_prompt(final_system, final_prompt),
        temperature=0,
    )
    output = response["choices"][0]["message"]["content"]
    return output


async def run_recursive_summarize(path):
    paths = sorted(list(os.listdir(path)))
    sub_file_summaries = {}
    sub_module_summaries = {}
    total_languages = set()
    file_tasks = []
    file_paths = []
    dir_tasks = []
    dir_paths = []
    for entity in paths:
        real_path = os.path.join(path, entity)
        if os.path.isfile(real_path):
            if ignore_file(real_path):
                continue
            file_tasks.append(asyncio.create_task(file_summary(real_path)))
            file_paths.append(relative_module(real_path))
            # sub_file_summaries[relative_module(real_path)] = result["summary"]
            # total_languages.add(result["language"])
        elif not ignore_dir(real_path):
            dir_tasks.append(asyncio.create_task(dir_summary(real_path)))
            dir_paths.append(relative_module(real_path))
            # sub_module_summaries[relative_module(real_path)] = result["summary"]
            # total_languages.add(result["language"])
    if (len(file_tasks) + len(dir_tasks)) == 0:
        return {"summary": "", "language": ""}
    results = await asyncio.gather(
        asyncio.gather(*file_tasks), asyncio.gather(*dir_tasks)
    )
    file_results = results[0]
    dir_results = results[1]
    for i in range(len(file_results)):
        sub_file_summaries[file_paths[i]] = file_results[i]["summary"]
        total_languages.add(file_results[i]["language"])
    for i in range(len(dir_results)):
        sub_module_summaries[dir_paths[i]] = dir_results[i]["summary"]
        total_languages.add(dir_results[i]["language"])
    language = " ".join(total_languages)

    # fast forward for single file or module
    if len(sub_file_summaries) == 1 and len(sub_module_summaries) == 0:
        dir_result = {
            "summary": list(sub_file_summaries.values())[0],
            "language": language,
        }
    elif len(sub_file_summaries) == 0 and len(sub_module_summaries) == 1:
        dir_result = {
            "summary": list(sub_module_summaries.values())[0],
            "language": language,
        }
    else:
        module = relative_module(path)
        console.print(f"[bold green]DIR[/bold green] {module}")
        file_summaries = construct_summary_pair(sub_file_summaries)
        module_summaries = construct_summary_pair(sub_module_summaries)
        summary = await prompt_summary(
            language=language,
            file_summaries=file_summaries,
            module_summaries=module_summaries,
            max_length=500,
            path=module,
        )
        dir_result = {"summary": summary, "language": language}
    return dir_result


async def dir_summary(path):
    module = relative_module(path)
    console.print(f"[bold green]DIR[/bold green] {module}")

    if envs["cache"] is not None:
        dir_cache = envs["cache"].get(module, None)
        if dir_cache is not None and dir_cache["hash"] == hash_dir(path):
            dir_result = dict(
                language=dir_cache["language"], summary=dir_cache["summary"]
            )
            console.rule(f"✓ {module}")
            return dir_result
    dir_result = await run_recursive_summarize(path)

    if envs["cache"] is not None and dir_result['summary'] != "":
        envs["cache"][module] = {}
        envs["cache"][module]["summary"] = dir_result['summary']
        envs["cache"][module]["language"] = dir_result['language']
        envs["cache"][module]["hash"] = hash_dir(path)

    console.rule(f"✓ {module}")
    return dir_result
