import json

from autonomous import log
from autonomous.ai import OpenAI

from dmtoolkit.models.ttrpgobject import TTRPGObject


class Item(TTRPGObject):
    rarity:str = ""
    cost:int = 0
    duration:str = ""
    weight:int = 0
    
    _funcobj = {
        "name": "generate_item",
        "description": "creates Item data object",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The item's name",
                },
                "desc": {
                    "type": "string",
                    "description": "A physical description of the item",
                },
                "traits": {
                    "type": "array",
                    "description": "A list of stats and special features of the item, if any.",
                    "items": {"type": "string"},
                },
                "weight": {
                    "type": "string",
                    "description": "The weight of the item",
                },
            },
        },
    }

    def get_image_prompt(self):
        description = f"{self.desc or self.name} in a display case"
        return f"A full color image in the style of Albrecht Dürer of an item called a {self.name}. Additional details:  {description}"

    @classmethod
    def generate(cls, world, description=None):
        primer = f"""
        As an expert AI in fictional {world.genre} worldbuilding, you generate items appropriate to the genre with a name and description.
        """
        prompt = f"Generate a random fictional {world.genre} loot item for a TTRPG with general stats. There is a 40% chance the item has an additional special feature or ability."
        if description:
            prompt += f" The item has the following description: {description}."
        obj_data = super().generate(primer, prompt)
        obj_data["world"] = world
        obj = cls(**obj_data)
        obj.save()
        return obj

    def page_data(self, root_path="ttrpg"):
        return {
            "Details": [
                f"rarity: {self.rarity if self.rarity else 'Unknown'}",
                f"cost: {self.cost if self.cost else 'Unknown'}",
                f"duration: {self.duration if self.duration else 'Unknown'}",
                f"weight: {self.weight if self.weight else 'Unknown'}",
            ]
        }
