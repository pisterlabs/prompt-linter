import requests
import re
from typing import Dict, Optional
from langchain.tools.base import BaseTool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
import pydash

class DND5E(BaseTool):

    """Tool that adds the capability to get information from the D&D 5th Edition API."""

    name = "DND5E"
    description = (
        "The preferred tool to use when you need to get information about the Dungeons and Dragons 5th Edition. "
        "Replies in JSON format. "
        "The Action Input should be a single string in the format of `resource/resource_name`. "
        "Accepted 'resource' items are: ['ability-scores', 'alignments', 'backgrounds', 'classes', 'conditions', 'damage-types', 'equipment-categories', 'equipment', 'feats', 'features', 'languages', 'magic-items', 'magic-schools', 'monsters', 'proficiencies', 'races', 'rules', 'rule-sections', 'skills', 'spells', 'subclasses', 'subraces', 'traits', 'weapon-properties']. "
        "The 'resource_name' is an optional free text query. "
        "Note: Querying for the same Action Input will always yield the same output."
    )
    acceptable_resources = ['ability-scores', 'alignments', 'backgrounds', 'classes', 'conditions', 'damage-types', 'equipment-categories', 'equipment', 'feats', 'features', 'languages', 'magic-items', 'magic-schools', 'monsters', 'proficiencies', 'races', 'rules', 'rule-sections', 'skills', 'spells', 'subclasses', 'subraces', 'traits', 'weapon-properties']

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the DND5E tool."""
        print("")
        print(f"==== DND5E qry: `{query}`")
        query = str.lower(query)
        match = re.match(r'(.*?)/(.*)|(.*)', query)
        if match.group(2):
            resource = match.group(1)
            resource_name = match.group(2)
            resource_name = re.sub("[ \+]", "-", resource_name)
        else:
            resource = match.group(0)
            resource_name = ""

        common_mistakes = {
            "weapons": "equipment",
            "armor": "equipment",
            "class": "classes",
            "items": "equipment",
        }
        resource = common_mistakes.get(resource, resource).replace(" ", "-")

        valid_resource = any(res == resource for res in self.acceptable_resources)
        
        if valid_resource == False:
            plural_check = resource[len(resource)-1] != "s"
            if plural_check:
                new_query = f"{resource}s"
                if resource_name:
                    new_query += f"/{resource_name}"
                return self._run(new_query)
            return f"Invalid Agent Input syntax ({query}), try again with Agent Input syntax formatted like `spell/acid arrow` but using your terms. Please verify your requested a resource is from the available list."

        url = f"https://www.dnd5eapi.co/api/{resource}"
        if resource_name:
            url += f"/{resource_name}"
        response = requests.get(url)
        if response.status_code == 200:
            raw_json = response.json()
            output = self.replace_url_keys(raw_json)
            return f"Information about {resource_name}:\n{output}"
        else:
            return f"Error getting {resource} information for {resource_name}: {response.status_code}"

    def replace_url_keys(self, original_json, new_json_object=None, parent_keypath=""):
        """Replaces all keys of "url" with a key of "DND5e Action Input", and modifies their value to string remove the "/api/" at any depth within the JSON object.

        Args:
            json_object: The JSON object to be modified.
            recursive: A boolean value that indicates whether the function should recursively iterate over the dictionary.

        Returns:
            The modified JSON object.
        """
        if new_json_object is None:
            new_json_object = original_json.copy()

        if parent_keypath == "":
            my_json_object = original_json
        else:
            my_json_object = pydash.get(original_json, parent_keypath[1:])

        to_create = {}
        to_delete = []

        if isinstance(my_json_object, dict):
            for key, value in my_json_object.items():
                current_keypath = f"{parent_keypath}.{key}"
                if isinstance(value, dict):
                    self.replace_url_keys(original_json, new_json_object, current_keypath)
                elif isinstance(value, list):
                    # for list_item in value:
                    for list_index in range(len(value)):
                        self.replace_url_keys(original_json, new_json_object, f"{current_keypath}[{list_index}]")
                elif key == "url":
                    found_keypath = current_keypath[1:]
                    my_keypath = re.sub(".url", ".DND5e Action Input", found_keypath)
                    new_value = value[len("/api/"):]

                    to_create[my_keypath] = new_value
                    to_delete.append(found_keypath)

        for add_key in to_create:
            pydash.update(new_json_object, add_key, to_create[add_key])

        for del_key in to_delete:
            pydash.unset(new_json_object, del_key)
            

        return new_json_object

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the DND5E tool asynchronously."""
        raise NotImplementedError("DND5E does not support async")
