import random
from dataclasses import dataclass
from pathlib import Path

import structlog
from guidance import Program
from guidance.llms import LLM

from wet_toast_talk_radio.common.dialogue import Line, Speaker, save_lines
from wet_toast_talk_radio.common.log_ctx import show_id_log_ctx
from wet_toast_talk_radio.media_store import MediaStore
from wet_toast_talk_radio.media_store.media_store import ShowId, ShowMetadata, ShowName
from wet_toast_talk_radio.scriptwriter.io import unique_script_filename
from wet_toast_talk_radio.scriptwriter.names import (
    GENDERS,
    random_name,
)
from wet_toast_talk_radio.scriptwriter.radio_show import RadioShow
from wet_toast_talk_radio.scriptwriter.the_expert_zone.missions import (
    random_host_missions,
)
from wet_toast_talk_radio.scriptwriter.the_expert_zone.titles import random_title
from wet_toast_talk_radio.scriptwriter.the_expert_zone.topics import random_topic
from wet_toast_talk_radio.scriptwriter.the_expert_zone.traits import random_trait

logger = structlog.get_logger()

AGENT_TEMPLATE = """
{{~#geneach 'conversation' stop=False}}
{{#system~}}
{{await 'system_message'}}
{{~/system}}{{#if first_question}}{{#assistant~}}
{{first_question}}
{{~/assistant}}{{/if}}
{{#user~}}
{{set 'this.question' (await 'question') hidden=False}}
{{~/user}}
{{#assistant~}}
{{gen 'this.response' temperature=0.9 max_tokens=250}}
{{~/assistant}}
{{~/geneach}}"""

# This doesn't require a llm, but using template language for clarity.
SYSTEM_MESSAGE_TEMPLATE = (
    "You are {{identity}}. You are {{role}}. {{mission}} "
    "You always answer in four sentences or less."
)


@dataclass
class Guest:
    name: str
    gender: str
    title: str
    trait: str
    topic: str
    # used to prevent biases in dialogue generation due to guest name
    placeholder_name: str


class TheExpertZone(RadioShow):
    """A radio show where expert guests are interviewed about their specialities."""

    def __init__(
        self,
        guest: Guest,
        host_missions: list[str],
        llm: LLM,
        media_store: MediaStore,
    ):
        self.guest = guest
        self.host_missions = host_missions
        self._llm = llm
        self._media_store = media_store

    @show_id_log_ctx()
    async def arun(self, show_id: ShowId) -> bool:
        lines = await self.agen()
        self._media_store.put_script_show(show_id=show_id, lines=lines)
        self._media_store.put_script_show_metadata(
            show_id=show_id, metadata=ShowMetadata(ShowName.THE_EXPERT_ZONE)
        )
        return True

    async def awrite(self, output_dir: Path) -> bool:
        lines = await self.agen()
        path = output_dir / unique_script_filename("the-expert-zone")
        save_lines(path=path, lines=lines)
        return True

    async def agen(self) -> list[Line]:
        logger.info("Async writing The Expert Zone")
        guest = Program(
            text=AGENT_TEMPLATE,
            first_question=None,
            llm=self._llm,
            async_mode=True,
            await_missing=True,
        )
        host = Program(
            text=AGENT_TEMPLATE,
            llm=self._llm,
            async_mode=True,
            await_missing=True,
        )

        # Intro
        guest_identity = (
            f"{self.guest.placeholder_name}, an expert researcher in {self.guest.topic}. "
            f"You are very {self.guest.trait}"
        )
        guest_role = "the guest on a talk show"
        guest_mission = "Do not repeat yourself throughout the conversation."
        system_message_prompt = Program(text=SYSTEM_MESSAGE_TEMPLATE, llm=self._llm)
        guest_system_message = system_message_prompt(
            identity=guest_identity, role=guest_role, mission=guest_mission
        )

        intro = (
            f"Welcome to 'The Expert Zone'! The show where we ask experts the difficult questions... "
            f"This is your star host, Nick. "
            f"Today we welcome {self.guest.placeholder_name}, {self.guest.title} {self.guest.topic}. "
            f"{self.guest.placeholder_name}, how are you?"
        )
        guest = await guest(system_message=guest_system_message, question=intro)

        host_identity = "Nick, a stupid and rude radio celebrity who hates their job"
        host_role = "the host of a talk show"

        # Body
        for i, host_mission in enumerate(self.host_missions):
            # first question needed for conversation context
            first_question = intro if i == 0 else None
            host_system_message = system_message_prompt(
                identity=host_identity, role=host_role, mission=host_mission
            )
            host = await host(
                first_question=first_question,
                system_message=host_system_message,
                question=guest["conversation"][-2]["response"],
            )
            guest = await guest(
                system_message=guest_system_message,
                question=host["conversation"][-2]["response"],
            )

        # Outro
        outro = (
            "Well, that's all the time we have today! "
            f"{self.guest.placeholder_name}, thank you for sharing your knowledge with us, "
            "and I wish you all the best in your groundbreaking research. "
            "Join us next time on 'The Expert Zone', where we ask the experts the difficult questions!"
        )
        guest = await guest(system_message=guest_system_message, question=outro)

        lines = self._post_processing(guest)
        logger.info("Finished writing The Expert Zone")
        return lines

    def _post_processing(self, program: Program) -> list[Line]:
        """Converts the guidance program into a list of Lines.
        Cleans up the content for each line."""
        logger.info("Post processing The Expert Zone")
        host = Speaker(name="Nick", gender="male", host=True)
        guest = Speaker(name=self.guest.name, gender=self.guest.gender, host=False)

        conversation = program["conversation"]
        lines = []
        # last exchange is always empty
        for exchange in conversation[:-1]:
            lines.append(
                Line(speaker=host, content=self._clean_content(exchange["question"]))
            )
            lines.append(
                Line(speaker=guest, content=self._clean_content(exchange["response"]))
            )
        return lines

    def _clean_content(self, content: str) -> str:
        """Replace placeholder names, remove new lines and extra whitespace."""
        content = content.replace(self.guest.placeholder_name, self.guest.name)
        content = " ".join(content.strip().split())
        return content

    @classmethod
    def create(cls, llm: LLM, media_store: MediaStore) -> "RadioShow":
        guest_gender = random.choice(GENDERS)
        guest_name = random_name(guest_gender)
        guest_placeholder_name = random_name(guest_gender)
        topic = random_topic()
        trait = random_trait()
        title = random_title()
        guest = Guest(
            name=guest_name,
            gender=guest_gender,
            title=title,
            trait=trait,
            topic=topic,
            placeholder_name=guest_placeholder_name,
        )
        logger.info("Random guest", guest=guest)

        host_missions = random_host_missions(topic=topic, k=8)
        logger.info("Random host missions", missions=host_missions)

        return cls(
            guest=guest,
            host_missions=host_missions,
            llm=llm,
            media_store=media_store,
        )
