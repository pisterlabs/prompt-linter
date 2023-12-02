#nlp/summarization.py
from openai import OpenAI
import json
from database import DatabaseManager
from database.models import Base, Character, Scene, SceneCharacter, Narrative, CharacterDescriptor
from config.settings import OPENAI_API_KEY, OPENAI_MODEL
from utils.utils import scene_to_text
class Summarizer:
    def __init__(self, db_manager):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.db_manager = db_manager

    def summarize_scene(self, current_scene: Scene):
        # Structure the messages for the chat completion
        messages = [
            {"role": "system", 
            "content": "You are a helpful assistant who summarizes scenes and character descriptions."},
            {"role": "user",
            "content": scene_to_text(current_scene)},
            {"role": "system",
            "content": "Please provide a brief summary of the scene described above."}
        ]

        # Call the OpenAI API
        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages
        )

        # Extract the summary from the response
        summary = response.choices[0].message.content.strip()

        # Update the narrative in the database with the summary
        narrative = self.db_manager.add_narrative(summary, current_scene)

        return narrative




if __name__ == "__main__":
    summarizer = Summarizer()
    summarized_narrative = summarizer.summarize_scene(scene_id=1)
    print(summarized_narrative.summary)
