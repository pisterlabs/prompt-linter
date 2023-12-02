from collections import deque
import json
from langchain.prompts import PromptTemplate

from environment.types import CrimeSceneMap
from llm.output_parsers.room import RoomJsonOutputParser


class RoomsChain:
    def __init__(self, llm, crime_scene_map: CrimeSceneMap) -> None:
        self._llm = llm
        self._crime_scene_map = crime_scene_map

        self._json_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["name", "description"],
        }

        self._example = {
            "theme": "Library of Alexandria, 340 BC, crazy librarian",
            "room": {
                "name": "The Ancient Manuscript Chamber",
                "description": "The room where Drusilla's lifeless body was discovered is known as the Ancient Manuscript Chamber. It is a dimly lit and secluded section of the library, hidden away from the bustling main halls. The chamber is lined with tall, dusty bookshelves filled with scrolls and manuscripts dating back centuries. The air is heavy with the scent of aging parchment and the mustiness of forgotten knowledge. Dim torches cast flickering shadows on the ancient tomes, creating an eerie ambiance. The room is a labyrinth of knowledge, and it was amidst this maze of scrolls and manuscripts that Drusilla met her tragic end, stabbed multiple times with an ancient dagger.",
            },
            "victim": {
                "name": "Archibald Ptolemy",
                "age": 35,
                "occupation": "Head Librarian",
                "murder_weapon": "Ancient scroll with poisoned ink",
                "death_description": "Found dead in his office surrounded by stacks of books, face contorted in a mixture of fear and surprise, as if he had been reading a particularly gruesome text when struck down.",
            },
            "suspect": {
                "name": "Cassandra",
                "age": 28,
                "occupation": "Junior Librarian",
                "alibi": "Cassandra asserts she was searching for lost texts in the stacks during the homicide. Multiple patrons corroborate her presence near the scene around the estimated time of death.",
            },
        }

        self._room_prompt = PromptTemplate.from_template(
            """
            <s>[INST] <<SYS>>
            
            You are a crime storyteller. Always output your answer in JSON using this scheme: {scheme}.
            
            <<SYS>>

            Given a theme: {theme_example} describe a room in the crime sceen.
            room:
            [/INST]
            {room_example}</s><s>
            
            [INST]
            Given a theme: {theme} describe a room in the crime sceen.
            room:
            [/INST]
            """
        )

        self._suspect_room_prompt = PromptTemplate.from_template(
            """
            <s>[INST] <<SYS>>
            
            You are a crime storyteller. Always output your answer in JSON using this scheme: {scheme}.
            
            <<SYS>>

            Given a theme: {theme_example}, suspect information: {entity_example}, describe a room where the suspect is in.
            room:
            [/INST]
            {room_example}</s><s>
            
            [INST]
            Given a theme: {theme}, suspect information: {entity}, describe a room where the suspect is in.
            room:
            [/INST]
            """
        )

        self._victim_room_prompt = PromptTemplate.from_template(
            """
            <s>[INST] <<SYS>>
            
            You are a crime storyteller. Always output your answer in JSON using this scheme: {scheme}.
            
            <<SYS>>

            Given a theme: {theme_example}, victim information: {entity_example}, describe a room where the victim's body was found.
            room:
            [/INST]
            {room_example}</s><s>
            
            [INST]
            Given a theme: {theme}, victim information: {entity}, describe a room where the victim's body was found.
            room:
            [/INST]
            """
        )

    def create(self, theme, victim, suspects):
        # generate description for the starting room, assumes square rooms layout
        middle_row, middle_col = (
            self._crime_scene_map.number_of_rooms,
            self._crime_scene_map.number_of_rooms,
        )
        start_story = self._generate_room(
            self._victim_room_prompt, theme, json.dumps(victim)
        )
        start_story.update({"row": middle_row, "col": middle_col})

        # contains final description of rooms
        rooms_data, suspects_positions = [start_story], []

        # generate rest of the rooms
        not_selected_suspects = deque(suspects)
        generated = set([(middle_row, middle_col)])
        q = deque(self._crime_scene_map.get_neighbours(middle_row, middle_col))
        while q:
            current_room = q.pop()
            if current_room in generated:
                continue

            row, col = current_room

            if not_selected_suspects:
                # there are suspects left
                current_room_story = self._generate_room(
                    self._suspect_room_prompt,
                    theme,
                    json.dumps(not_selected_suspects.popleft()),
                )
                suspects_positions.append({"row": row, "col": col})
            else:
                current_room_story = self._generate_room(self._room_prompt, theme)

            current_room_story.update({"row": row, "col": col})

            rooms_data.append(current_room_story)
            generated.add((row, col))
            q.extend(self._crime_scene_map.get_neighbours(row, col))

        return rooms_data, suspects_positions

    def _generate_room(self, prompt, theme, entity=None):
        chain = prompt | self._llm | RoomJsonOutputParser()

        input_data = {
            "room_example": json.dumps(self._example["room"]),
            "scheme": json.dumps(self._json_schema),
            "theme": theme,
            "theme_example": self._example["theme"],
        }
        if entity:
            input_data.update({
                "entity": json.dumps(entity),
                "entity_example": json.dumps(
                    self._example["victim"]
                    if prompt == self._victim_room_prompt
                    else self._example["suspect"]
                ),
            })

        return chain.invoke(input_data)
