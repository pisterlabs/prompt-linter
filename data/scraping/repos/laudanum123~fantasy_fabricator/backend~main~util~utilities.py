"""Different utilities for the backend"""
import json
import re
import openai
from main import db
from main.models import AdventureNPCs, AdventureLocations



def clean_gpt_response(gpt_response: str, expected_keys: list) -> dict:
    """
    clean gpt response
    """
    # Remove leading/trailing whitespaces
    gpt_response = gpt_response.strip()

    gpt_response = re.sub(r"[^\x20-\x7E]+", "", gpt_response)
    # Remove extra quotation marks
    # gpt_response = re.sub('"', '', gpt_response)

    gpt_response = verify_gpt_response_keys(gpt_response, expected_keys)
    # Parse response into a Python dictionary
    response_dict = json.loads(gpt_response)

    # Verify keys and make necessary adjustments
    for key in expected_keys:
        if key not in response_dict:
            response_dict[key] = ""

    # Turn the JSON string into a JSON response
    return response_dict


def verify_gpt_response_keys(response_string: str, expected_keys: list) -> str:
    """verify that the keys in the response string are correct

    Args:
        response_string (str): pre_cleaned response string

    Returns:
        str: cleaned response string
    """

    # Split after each key-value pair and then split key and value
    # Replace incorrect key with correct one if necessary
    key_value_pairs = response_string.split('",')
    print(key_value_pairs)
    for i, pair in enumerate(key_value_pairs):
        key = pair[: pair.index(":")]
        value = pair[pair.index(":") + 1 :]
        if key.strip() != expected_keys[i]:
            key_value_pairs[i] = f'"{expected_keys[i]}": {value}'

    # Rejoin key-value pairs and add curly braces if necessary
    new_response = '",'.join(key_value_pairs)
    if new_response[0] != "{":
        new_response = "{" + new_response
    if new_response[-1] != "}":
        new_response = new_response + "}"

    return new_response


def extract_entities_from_adventure(adventure: list):

    adv_id = adventure["id"]
    adventure.pop("id", None)

    # join all the values of the dictionary into one string
    corpus = " ".join(adventure.values())
    prompt = f'Given the following RPG story: {corpus}.\
    Extract all entities. \
    Entities refer to all NPCs (non-player character) and locations.\
    NPCs include but not limited to: Any character mentioned in the story such as NPCs, side characters,players,characters and living beings\
    Locations include but not limited to: Any location mentioned in the RPG story such as Forest,Desert,River,Oceans,Temple. \
    The extracted content should exactly string match what is present in the RPG story. \
    Format the answer as a json object with the following structure with a value of type "list":\
    {{  "NPCs": [NPC content],\
        "Locations": [Locations content] }}'

    response = openai.Completion.create(
        engine="text-davinci-003", prompt=prompt, max_tokens=2000
    )

    text = response["choices"][0]["text"]

    npc_loc_json = json.loads(text)

    npc_list = npc_loc_json.get("NPCs")
    locations_list = npc_loc_json.get("Locations")

    for npc in npc_list:
        npc_obj = AdventureNPCs.query.filter_by(
            adventure_id=adv_id, npc_name=npc
        ).first()
        if not npc_obj:
            npc_obj = AdventureNPCs(adventure_id=adv_id, npc_name=npc)
            db.session.add(npc_obj)

    for location in locations_list:
        location_obj = AdventureLocations.query.filter_by(
            adventure_id=adv_id, location_name=location
        ).first()
        if not location_obj:
            location_obj = AdventureLocations(
                adventure_id=adv_id, location_name=location
            )
            db.session.add(location_obj)

    db.session.commit()
    return npc_list, locations_list
