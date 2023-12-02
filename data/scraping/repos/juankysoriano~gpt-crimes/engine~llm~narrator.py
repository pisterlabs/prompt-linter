import os

import langchain.output_parsers.json as json
import openai
from langchain.chat_models import ChatOpenAI

from engine.llm.templates import CONTINUE_STORY_TEMPLATE, START_STORY_TEMPLATE


class _Narrator:
    def __init__(self):
        pass

    async def new_story(
        self,
    ):
        llm = ChatOpenAI(
            verbose=True,
            model="gpt-4",
            temperature=0.5,
        )
        result = await llm.apredict(
            text=START_STORY_TEMPLATE,
        )
        result = json.parse_and_check_json_markdown(
            result, expected_keys=["text", "image", "available_actions"]
        )
        result["image_url"] = self._image(prompt=result["image"])
        return result

    async def next_scene(
        self, history, last_action, scene_number, total_scenes
    ):
        llm = ChatOpenAI(
            verbose=True,
            model="gpt-4",
            temperature=0.5,
        )
        result = await llm.apredict(
            text=CONTINUE_STORY_TEMPLATE.format(
                history=history,
                last_action=last_action,
                percentage=round((scene_number / total_scenes) * 100),
            ),
        )
        result = json.parse_and_check_json_markdown(
            result, expected_keys=["text", "image", "available_actions"]
        )
        result["image_url"] = self._image(prompt=result["image"])
        return result

    def _image(self, prompt):
        result = openai.Image.create(
            prompt=f"{prompt}. Black and White. The Walking Dead Comic Style. ",
            n=1,
            size="512x512",
        )
        return result["data"][0]["url"]
