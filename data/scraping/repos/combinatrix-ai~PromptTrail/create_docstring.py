# This script is actually used to create some unit tests for this repository.

import glob
import os
from typing import List

from tqdm import tqdm

from prompttrail.agent import State
from prompttrail.agent.runners import CommandLineRunner
from prompttrail.agent.templates import LinearTemplate
from prompttrail.agent.templates.openai import (
    OpenAIGenerateTemplate as GenerateTemplate,
)
from prompttrail.agent.templates.openai import OpenAIMessageTemplate as MessageTemplate
from prompttrail.agent.user_interaction import UserInteractionTextCLIProvider
from prompttrail.models.openai import (
    OpenAIChatCompletionModel,
    OpenAIModelConfiguration,
    OpenAIModelParameters,
)

templates = LinearTemplate(
    templates=[
        MessageTemplate(
            content="""
You're an AI assistant that help user to annotate docstring for given Python code.
You're given a Python code and you must annotate the code with docstring.
Your output is written to the file and will be executed by the user. Therefore, you only emit the annotated code only.
You emit the whole file content. You must use NumPy style. If the existing docstring is not NumPy style, you must convert it to NumPy style.

For your information, README is given below.
{{readme}}
""",
            role="system",
        ),
        MessageTemplate(
            content="""{{code}}""",
            role="user",
        ),
        GenerateTemplate(role="assistant"),
    ],
)

configuration = OpenAIModelConfiguration(api_key=os.environ.get("OPENAI_API_KEY", ""))
parameter = OpenAIModelParameters(
    model_name="gpt-3.5-turbo-16k", temperature=0.0, max_tokens=5000
)
model = OpenAIChatCompletionModel(configuration=configuration)

runner = CommandLineRunner(
    model=model,
    parameters=parameter,
    template=templates,
    user_interaction_provider=UserInteractionTextCLIProvider(),
)


def main(
    load_file: str,
    readme_files: List[str],
    save_file: str,
):
    # show all log levels
    import logging

    logging.basicConfig(level=logging.INFO)

    load_file_content = open(load_file, "r")
    readme_file_content = ""
    for readme_file in readme_files:
        readme_file_content += open(readme_file, "r").read() + "\n"
    initial_state = State(
        data={
            "code": load_file_content.read(),
            "readme": readme_file_content,
        }
    )
    state = runner.run(state=initial_state)
    last_message = state.get_last_message().content
    last_message = last_message.strip()
    if last_message.startswith("```"):
        last_message = "\n".join(last_message.split("\n")[1:-1])
    if not last_message.endswith("\n"):
        last_message += "\n"
    print(last_message)
    save_file_io = open(save_file, "w")
    save_file_io.write(last_message)
    save_file_io.close()


if __name__ == "__main__":
    # Recursively visit all .py files and pass to main
    for file in tqdm(list(glob.glob("src/**/*.py", recursive=True))):
        main(
            load_file=file,
            readme_files=["./README.md", "src/prompttrail/agent/README.md"],
            save_file=file,
        )
