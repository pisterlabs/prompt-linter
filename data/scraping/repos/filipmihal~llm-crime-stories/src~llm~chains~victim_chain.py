import json
from langchain.prompts import PromptTemplate

from llm.output_parsers.victim import VictimJsonOutputParser


class VictimChain:
    def __init__(self, llm):
        self._json_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "occupation": {"type": "string"},
                "murder_weapon": {"type": "string"},
                "death_description": {"type": "string"},
            },
            "required": [
                "name",
                "age",
                "occupation",
                "murder_weapon",
                "death_description",
            ],
        }

        self._example = {
            "theme": "Library of Alexandria, 340 BC, crazy librarian",
            "victim": {
                "name": "Archibald Ptolemy",
                "age": 42,
                "occupation": "Head librarian",
                "murder_weapon": "fragile ancient scroll",
                "death_description": "Found face down under pile of books with a broken quill pen lodged in his back, surrounded by scattered papyrus rolls.",
            }
        }

        prompt = PromptTemplate.from_template(
            """
            <s>[INST] <<SYS>>
            
            You are a crime storyteller. Always output answer as JSON using this {scheme}.
                        
            <<SYS>>

            Given a theme: {theme_example}. describe a victim of the story. Avoid nicknames.
            victim:
            [/INST]
            {victim_example}</s><s>
            
            [INST]
            Given a theme: {theme} describe a victim of the story. Avoid nicknames.
            victim:
            [/INST]
            """
        )

        self._chain = prompt | llm | VictimJsonOutputParser()

    def create(self, theme):
        return self._chain.invoke(
            {
                "scheme": json.dumps(self._json_schema),
                "theme": theme,
                "theme_example": self._example["theme"],
                "victim_example": json.dumps(self._example["victim"]),
            }
        )
