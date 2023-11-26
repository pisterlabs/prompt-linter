#!/usr/bin/env python

import datetime
import pathlib
import typing
from openai import OpenAI


API_KEY = ""
MODEL = "gpt-3.5-turbo-16k"  # or gpt-4-32k
TEMPERATURE = 0.2
DIRECTORY = pathlib.Path.cwd() / "project"
ALLOW_LIST = [
    # extensions
    ".c",
    ".cpp",
    ".env",
    ".h",
    ".json",
    ".py",
    ".sh",
    ".toml",
    ".xml",
    ".yaml",
    ".yml",
    # other files
    "CMakeLists.txt",
    "package.xml",
    "LICENSE",
]


def get_timestamp() -> typing.AnyStr:
    return datetime.datetime.now().strftime("%Y_%m_%dT%H_%M_%S")


def generate_prompt(
    directory: pathlib.Path, allow_list: typing.List[typing.AnyStr] = None
) -> typing.AnyStr:
    with open("preprompt.md", "r") as preprompt_file:
        prompt = preprompt_file.read()
    prompt += "\n\nProject files:\n\n"
    directory_path = pathlib.Path(directory)
    for path in directory_path.glob("**/*"):
        if path.is_file():
            file_extension = path.suffix.lower()
            if file_extension in allow_list or path.name in allow_list:
                with open(path, "r") as source_file:
                    prompt += "- `" + str(path.relative_to(directory_path)) + "`:\n"
                    for line in source_file.readlines():
                        prompt += "    " + line
                    prompt += "\n"
    with open(f"prompt-{get_timestamp()}.md", "w") as prompt_file:
        prompt_file.write(prompt)
    return prompt


def main():
    client = OpenAI(api_key=API_KEY)
    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[
            {
                "role": "user",
                "content": f"{generate_prompt(DIRECTORY, ALLOW_LIST)}",
            },
        ],
    )
    with open(f"response-{get_timestamp()}.md", "w") as response_file:
        response_file.write(response.choices[0].message.content)


if __name__ == "__main__":
    main()
