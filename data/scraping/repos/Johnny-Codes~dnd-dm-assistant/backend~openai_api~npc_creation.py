import os
import json
import re
import openai

from .prompts import (
    chat_roles,
    ai_models,
    output_content_type,
)


def npc_creation(work, add_info):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.ChatCompletion.create(
        model=ai_models["3_5"],
        messages=[
            {"role": "system", "content": chat_roles["initial"]},
            {
                "role": "user",
                "content": f"I need you to create a character that works at {work}. They are not an important npc. I need a name, race, personality, physical description, and 2 or 3 role playing tips. {add_info}"
                + " "
                + output_content_type["npc_level_one_dict"],
            },
        ],
        max_tokens=512,
        temperature=0.9,
    )
    if response.choices[0].finish_reason == "stop":
        character = response.choices[0].message.content

        character_dict_match = re.search(r"{[^}]+}", character)
        if character_dict_match:
            character_json_str = character_dict_match.group(0)
            character_j = json.loads(character_json_str)
            return character_j
        else:
            return None
    else:
        npc_creation(work, add_info)
