import json
import random

from autonomous import log
from autonomous.ai import OpenAI

from dmtoolkit.models.base.encounter import Encounter


class DnDEncounter(Encounter):
    @classmethod
    def generate(cls, num_players=5, level=1):
        primer = """
        You are an expert D&D5e Encounter generator that creates level appropriate random encounters and specific loot rewards.
        """
        enc_type = "D&D5e"
        encounter = super().generate(primer=primer, enc_type=enc_type)
        return DnDEncounter(**encounter.__dict__)
