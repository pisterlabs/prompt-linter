import os
from pprint import pprint

from openai_client import get_gpt_response
from prompt_builder import PromptBuilder
from utils import get_tmp_env_var_name, typer_writer

prompt_builder = PromptBuilder()
ACTIONS = [
    "SHOW_ENV_VARS", "SET_ENV_VAR", "DELETE_ENV_VAR",
    "LOAD_FILE", "SAVE_FILE",
]


def handle_shell_action(inp, env_var2val, latest_response) -> bool:
    inp = inp.replace("DO: ", "")
    prompt = prompt_builder.do_prompt(inp, actions=ACTIONS)
    response = get_gpt_response(
        prompt, temperature=1, top_p=1, caching=False, chat=None)
    if not response.startswith("ACTION: "):
        print(response)
        typer_writer(
            "The response is not a valid action. This is a bug from OpenAI.")
        return False

    response = response.replace("ACTION: ", "")
    if "," in response or response not in ACTIONS:
        typer_writer(
            "Your input leads to multiple possible actions. Please be more specific.")
        typer_writer(f"Available actions: {ACTIONS}",)
        return False

    action_name = response.replace("ACTION: ", "")
    if action_name == "LOAD_FILE":
        return handle_load_file(inp, env_var2val)
    elif action_name == "SAVE_FILE":
        return handle_save_file(inp, env_var2val)
    elif action_name == "SHOW_ENV_VARS":
        return handle_show_env_vars(inp, env_var2val)
    elif action_name == "SET_ENV_VAR":
        return handle_set_env_var(inp, env_var2val, latest_response)
    elif action_name == "DELETE_ENV_VAR":
        return handle_delete_env_var(inp, env_var2val)
    else:
        print(f"ACTION: {action_name} is not implemented yet.")
        return False


def handle_set_env_var(inp, env_var2val, latest_response) -> bool:
    get_var_name_prompt = prompt_builder.set_env_var_get_var_name_prompt(inp)
    get_var_name_response = get_gpt_response(
        get_var_name_prompt, temperature=1, top_p=1, caching=False, chat=None)
    get_var_name_response = get_var_name_response.replace("Output: ", "")
    if not get_var_name_response.startswith("VAR_NAME: "):
        typer_writer(get_var_name_response)
        typer_writer(
            "Cannot identify variable name. This is a bug from OpenAI.")
        return False
    to_var_name = get_var_name_response.replace("VAR_NAME: ", "").strip()
    get_content_prompt = prompt_builder.set_env_var_get_content_prompt(inp)
    get_content_response = get_gpt_response(
        get_content_prompt, temperature=1, top_p=1, caching=False, chat=None)
    get_content_response = get_content_response.replace("Output: ", "")
    if "LAST_RESPONSE" in get_content_response:
        env_var2val[to_var_name] = latest_response
    elif "VAR_NAME: " in get_content_response:
        from_var_name = get_content_response.replace(
            "VAR_NAME: ", "").strip()
        if from_var_name not in env_var2val:
            typer_writer(
                f"Variable {from_var_name} is not defined. Please define it first.")
            return False
        env_var2val[to_var_name] = env_var2val[from_var_name]
    elif "VALUE: " in get_content_prompt:
        env_var2val[to_var_name] = get_content_response.replace(
            "VALUE: ", "").strip()
    else:
        typer_writer(
            "Cannot identify the value to be set. This is a bug from OpenAI.")
        return False
    typer_writer(f"Setting the variable with name {to_var_name}")
    return str(env_var2val[to_var_name])


def handle_delete_env_var(inp, env_var2val) -> bool:
    prompt = prompt_builder.delete_env_var_prompt(inp)
    response = get_gpt_response(
        prompt, temperature=1, top_p=1, caching=False, chat=None)
    if not response.startswith("VAR_NAMES:"):
        typer_writer(response)
        typer_writer(
            "The response is not a valid action. This is a bug from OpenAI.")
        return False
    response = response.replace("VAR_NAMES:", "").strip()
    var_names = response.split(",")
    if len(var_names) == 1 and len(var_names[0]) == 0:
        var_names = list(env_var2val.keys())
    for var_name in var_names:
        var_name = var_name.strip()
        if var_name not in env_var2val:
            typer_writer(f"Variable {var_name} is not defined. Skipping...")
            continue
        del env_var2val[var_name]
        typer_writer(f"Variable {var_name} deleted.")
    return True


def handle_load_file(inp, env_var2val) -> bool:
    prompt = prompt_builder.load_file_prompt(inp)
    response = get_gpt_response(
        prompt, temperature=1, top_p=1, caching=False, chat=None)
    if not response.startswith("FILE_PATHS: "):
        typer_writer(response)
        typer_writer(
            "The response is not a valid action. This is a bug from OpenAI.")
        return False
    response = response.replace("FILE_PATHS: ", "")
    file_paths = response.split(",")

    for file_path in file_paths:
        file_path = file_path.strip()
        new_env_var_name = get_tmp_env_var_name(
            env_var2val, "FILE_CONTENT_VAR_")
        try:
            with open(file_path, "r") as fp:
                env_var2val[new_env_var_name] = fp.read()
        except FileNotFoundError:
            typer_writer(f"File {file_path} not found.")
            return False
        typer_writer(f"{file_path} loaded.")
    return True


def handle_save_file(inp, env_var2val) -> bool:
    prompt = prompt_builder.save_file_prompt(inp)
    response = get_gpt_response(
        prompt, temperature=1, top_p=1, caching=False, chat=None)
    if not "FILE_PATH: " in response and "VAR_NAME: " in response:
        typer_writer(response)
        typer_writer(
            "The response is not a valid action. This is a bug from OpenAI.")
        return False
    file_path, var_name = response.split(",")
    file_path = file_path.replace("FILE_PATH: ", "").strip()
    var_name = var_name.replace("VAR_NAME: ", "").strip()
    typer_writer(f"Saving {var_name} to {file_path}.")
    try:
        with open(file_path, "w+") as fp:
            fp.write(env_var2val[var_name])
        return True
    except FileNotFoundError:
        typer_writer(f"Path {file_path} not found.")
        typer_writer(f"Formatted prompt: {prompt}")
        typer_writer(f"Untouched response: {response}")
        return False


def handle_show_env_vars(inp, env_var2val) -> bool:
    if len(env_var2val) == 0:
        typer_writer("No environment variables loaded.")
        return True
    else:
        prompt = prompt_builder.show_env_vars_prompt(inp)
        response = get_gpt_response(
            prompt, temperature=1, top_p=1, caching=False, chat=None)
        if not response.startswith("ENV_VARS:"):
            typer_writer(response)
            typer_writer(
                "The response is not a valid action. This is a bug from OpenAI.")
            return False
        env_vars = response.replace("ENV_VARS:", "").split(",")

        show_full = True
        if "" in env_vars:
            env_vars.remove("")
        if len(env_vars) == 0:
            show_full = False
            env_vars = env_var2val.keys()

        if len(env_vars) == 0:
            typer_writer("No environment variables loaded.")
            return True

        for env_var in env_vars:
            val = env_var2val[env_var.strip()]
            if show_full:
                typer_writer(f"{env_var} = {val}")
            else:
                typer_writer(f"{env_var} = {val[:50] if val else None}" +
                             ("..." if val and len(val) > 50 else ""))
        return True
