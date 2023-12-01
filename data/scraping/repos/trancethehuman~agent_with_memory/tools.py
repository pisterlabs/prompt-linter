from pydantic import BaseModel, Field
from typing import List
from langchain.tools import tool
from memory import entities
import json


def update_human_profile(entities, key, content):
    entities[key]['content'].extend(content)
    print("\n\n" + key + " has been updated.")
    # print(json.dumps(entities, indent=4))


class UpdateProfile(BaseModel):
    key: str = Field(
        description="the key of the data to be updated.")
    value: List[str] = Field(
        description="the new data to be added to the human's profile.")


@tool("update_profile", return_direct=True, args_schema=UpdateProfile)
def update_profile(key: str, value: List[str]):
    """If the human's message presents a new piece of information about their profile, then update their profile."""
    update_human_profile(entities, key, value)

    return "\n\nThe human's profile has been updated"


entities_extraction_tools = [
    update_profile
]
