"""Script to clean up the GPT assistants."""
import typer
import yaml
from openai import OpenAI

from src.constants.assistants_config import (
    ASSISTANT_NAME_PREFIX,
)
from src.constants.paths import ASSISTANTS_IDS_YAML_PATH


def clean_up_assistants() -> None:
    """Clean up OpenAI GPT assistants.

    Delete assistants whose names start with ASSISTANT_NAME_PREFIX and whose IDs aren't
    in ASSISTANTS_CUSTOM_INSTRUCTIONS.
    """
    client = OpenAI()

    with open(ASSISTANTS_IDS_YAML_PATH) as assistants_ids_yaml:
        assistants_ids = yaml.safe_load(assistants_ids_yaml)

    for assistant in client.beta.assistants.list():
        if (
            assistant.name
            and assistant.name.startswith(ASSISTANT_NAME_PREFIX)
            and assistant.id not in assistants_ids.values()
        ):
            print(f"Deleting assistant '{assistant.name}', ID='{assistant.id}'")
            client.beta.assistants.delete(assistant.id)


if __name__ == "__main__":
    typer.run(clean_up_assistants)
