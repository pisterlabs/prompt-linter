import subprocess

import openai
from openai.error import APIConnectionError


def generate_commit_messages(
    api_key: str, prompt: str, language: str = "english", num_messages: int = 5
) -> list:
    try:
        openai.api_key = api_key
        model_engine = "text-davinci-002"
        prompt = f"What follows '-------' is a git diff for a potential commit. Reply with an appropriate git commit message(a Git commit message should be concise but also try to describe the important changes in the commit) and don't include any other text but the message in your response. ------- {prompt}, language={language}"
        completions = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=200,
            n=num_messages,
            stop=None,
            temperature=0.7,
        )
        messages = [c.text.strip().replace("\n", "") for c in completions.choices]
        return messages
    except APIConnectionError as e:
        error_message = f"OpenAI API Error: {e}"
        print(error_message)
        raise openai.APIError(error_message)


def get_diff(diff_per_file: bool) -> str:
    if diff_per_file:
        diff = subprocess.check_output(
            "git diff --cached --name-only",
            shell=True,
            stderr=subprocess.STDOUT,
        ).decode("utf-8")

        files_changed = diff.split("\n")[:-1]
        diff_string = ""
        for file in files_changed:
            diff_string += subprocess.check_output(
                f"git diff --cached -- {file}",
                shell=True,
                stderr=subprocess.STDOUT,
            ).decode("utf-8")
    else:
        diff_string = subprocess.check_output(
            "git diff --cached .",
            shell=True,
            stderr=subprocess.STDOUT,
        ).decode("utf-8")

    return diff_string
