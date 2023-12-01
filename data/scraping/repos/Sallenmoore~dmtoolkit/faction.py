import json
import random

from autonomous import log
from autonomous.ai import OpenAI

from dmtoolkit.models.ttrpgobject import TTRPGObject

from .character import Character


class Faction(TTRPGObject):
    goal: str = ""
    status: str = ""
    leader: Character = None
    members: list[Character] = []

    _personality = {
            "social": [
                "secretive",
                "agressive",
                "courageous",
                "cowardly",
                "quirky",
                "imaginative",
                "reckless",
                "cautious",
                "suspicious",
                "friendly",
                "unfriendly",
            ],
            "political": [
                "practical",
                "violent",
                "cautious",
                "sinister",
                "anarchic",
                "religous",
                "patriotic",
                "nationalistic",
                "xenophobic",
                "racist",
                "egalitarian",
            ],
            "economic": [
                "disruptive",
                "ambitious",
                "corrupt",
                "charitable",
                "greedy",
                "generous",
            ],
        }

    _funcobj = {
        "name": "generate_faction",
        "description": "completes Faction data object",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The faction's name",
                },
                "traits": {
                    "type": "array",
                    "description": "The faction's personality",
                    "items": {"type": "string"},
                },
                "desc": {
                    "type": "string",
                    "description": "A description of the members of the faction",
                },
                "backstory": {
                    "type": "string",
                    "description": "The faction's backstory",
                },
                "goal": {
                    "type": "string",
                    "description": "The faction's goal",
                },
                "status": {
                    "type": "string",
                    "description": "The faction's current status",
                },
            },
        },
    }

    @classmethod
    def generate(cls, world, description=None):
        primer = f"""
        You are an expert {world.genre} TTRPG Worldbuilding AI that generates interesting random factions and organizations for a TTRPG."
        """
        traits = [random.choice(traits) for traits in cls._personality.values()]
        prompt = f"Generate a {world.genre} faction with the following traits: {', '.join(traits)} for a TTRPG in a location described as follows: \n{description}.  The faction needs a backstory containing an unusual, wonderful, OR sinister secret that gives the Faction a goal they are working toward."
        obj_data = super().generate(primer, prompt)
        obj_data |= {"world": world, "traits": traits}
        obj = cls(**obj_data)
        obj.save()
        return obj

    def get_image_prompt(self):
        return f"A full color logo or banner for a fictional faction named {self.name} and described as {self.desc}."

    def add_member(self, character=None, leader=False):
        description = ""
        if leader:
            description = (
                f"The leader of the {self.name} faction whose goal is: {self.goal}."
            )
        if not character:
            character = Character.generate(self.world, description)
            character.save()

        if character not in self.members:
            self.members.append(character)

        if leader:
            self.leader = character

        self.save()
        return character

    def page_data(self, root_path="ttrpg"):
        members = (
            [f"{m.name}]({m.wiki_path})" for m in self.members]
            if self.members
            else "Unknown"
        )
        return {
            "Goal": self.goal,
            "Leader": f"{self.leader.name}]({self.leader.wiki_path})"
            if self.leader
            else "Unknown",
            "Details": [
                f"status: {self.status if self.status else 'Unknown'}",
                f"members: {members}",
            ],
        }

    def canonize(self, api=None, root_path="ttrpg"):
        if not api:
            api = self._wiki_api
        super().canonize(api, root_path)
        for f in self.members:
            f.canonize(api=api, root_path=root_path)
        self.save()
