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
from wet_toast_talk_radio.scriptwriter.prolove.anecdotes import random_anecdote
from wet_toast_talk_radio.scriptwriter.prolove.conversation import History, Role
from wet_toast_talk_radio.scriptwriter.prolove.genders import Gender
from wet_toast_talk_radio.scriptwriter.prolove.host_messages import random_host_messages
from wet_toast_talk_radio.scriptwriter.prolove.lessons import random_lesson
from wet_toast_talk_radio.scriptwriter.prolove.missions import (
    GuestMissions,
    HostMissions,
)
from wet_toast_talk_radio.scriptwriter.prolove.products import random_product
from wet_toast_talk_radio.scriptwriter.prolove.sexual_orientations import (
    random_sexual_orientation,
)
from wet_toast_talk_radio.scriptwriter.prolove.topics import random_topic
from wet_toast_talk_radio.scriptwriter.prolove.traits import random_trait
from wet_toast_talk_radio.scriptwriter.radio_show import RadioShow

logger = structlog.get_logger()

# Not using geneach to have more control of conversation history
# More guided by conversation
TOP_SYSTEM_TEMPLATE = """
{{#system~}}
{{system_message}}
{{~/system}}
{{~#each history}}
{{#role this.role~}}
{{this.message}}{{~/role}}{{/each}}
{{#assistant~}}
{{gen 'response' temperature=0.7 max_tokens=800}}
{{~/assistant}}"""

# More guided by system message
BOTTOM_SYSTEM_TEMPLATE = """
{{~#each history}}
{{#role this.role~}}
{{this.message}}{{~/role}}{{/each}}
{{#system~}}
{{system_message}}
{{~/system}}
{{#assistant~}}
{{gen 'response' temperature=0.7 max_tokens=500}}
{{~/assistant}}"""

SYSTEM_MESSAGE_TEMPLATE = "{{identity}} {{role}} {{mission}} "

HOST_IDENTITY = (
    "You are Zara, a sassy and confident dating coach who unintentionally gives terrible advice. "
    "You speak in a friendly and informal manner. "
)

HOST_ROLE = "You are the host of a radio talk show."

GUEST_IDENTITY = (
    "Your name is {{name}}, and you are {{age}} years old. "
    "You are {{gender}} and you are {{sexual_orientation}}. "
    "You are very {{trait}}, and you speak in a friendly and informal manner."
)
GUEST_ROLE = "You are the guest on a talk show."


@dataclass
class Guest:
    name: str
    gender: Gender
    age: int
    sexual_orientation: str
    voice_gender: str
    trait: str
    topic: str
    # used to prevent biases in dialogue generation due to guest name
    placeholder_name: str


class Prolove(RadioShow):
    """A dating advice radio show."""

    def __init__(  # noqa: PLR0913
        self,
        guest: Guest,
        host_missions: HostMissions,
        guest_missions: GuestMissions,
        host_messages: list[str],
        llm: LLM,
        media_store: MediaStore,
    ):
        self.guest = guest
        self.host_missions = host_missions
        self.guest_missions = guest_missions
        self.host_messages_buffer = host_messages
        self._llm = llm
        self._media_store = media_store

    @show_id_log_ctx()
    async def arun(self, show_id: ShowId) -> bool:
        lines = await self.agen()
        self._media_store.put_script_show(show_id=show_id, lines=lines)
        self._media_store.put_script_show_metadata(
            show_id=show_id, metadata=ShowMetadata(ShowName.PROLOVE)
        )
        return True

    async def awrite(self, output_dir: Path) -> bool:
        lines = await self.agen()
        path = output_dir / unique_script_filename("prolove")
        save_lines(path=path, lines=lines)
        return True

    async def agen(self) -> list[Line]:
        logger.info("Async writing Prolove")

        # PART 1
        # The host introduces the show and themselves, then tells an anecdote about their dating life.

        intro_1 = "Hey there, my hearts! Welcome to 'Prolove', the dating advice show where we LOVE listening to you! "
        intro_2 = "I'm your host, Zara, ready to answer all your questions about love and connection. "
        history_pt1 = History()
        history_pt1.append(role=Role.HOST, message=intro_1)
        history_pt1.append(role=Role.HOST, message=intro_2)

        for mission in self.host_missions.pt1_missions:
            host = await self.new_top_program()(
                system_message=self.host_system_message(mission),
                history=history_pt1.host_history,
            )
            history_pt1.append(role=Role.HOST, message=host["response"])

        # PART 2
        # A caller is on the line. They explain their concern and the host gives them advice.

        # Don't keep any messages from part 1
        history_pt2 = History()
        for host_mission, guest_mission in zip(
            self.host_missions.pt2_missions,
            self.guest_missions.pt2_missions,
            strict=True,
        ):
            if host_mission is None:
                message = self.host_messages_buffer.pop(0)
            else:
                host = await self.new_top_program()(
                    system_message=self.host_system_message(host_mission),
                    history=history_pt2.host_history,
                )
                message = host["response"]
            history_pt2.append(role=Role.HOST, message=message)

            guest = await self.new_top_program()(
                system_message=self.guest_system_message(guest_mission),
                history=history_pt2.guest_history,
            )
            history_pt2.append(role=Role.GUEST, message=guest["response"])

        # PART 3
        # The host changes topics and chats about random things.

        history_pt3 = History(history_pt2.messages[-2:])
        for host_mission, guest_message_pt3 in zip(
            self.host_missions.pt3_missions,
            self.guest_missions.pt3_messages,
            strict=True,
        ):
            host = await self.new_bottom_program()(
                # Don't mention that Zara is host or she keeps asking questions
                system_message=self.host_system_message(host_mission, role=""),
                history=history_pt3.host_history,
            )
            history_pt3.append(role=Role.HOST, message=host["response"])
            history_pt3.append(role=Role.GUEST, message=guest_message_pt3)

        outro = (
            "Well, my hearts, that's all the time we have today! "
            f"Thanks again for calling, {self.guest.placeholder_name}, much love to you, "
            "and to all my hearts, remember to say YES to love, and see you next time! "
        )
        history_pt3.append(role=Role.HOST, message=outro)

        # Collect all messages
        conversation = (
            history_pt1.messages + history_pt2.messages + history_pt3.messages[2:]
        )
        lines = self._post_processing(conversation)
        logger.info("Finished writing Prolove")
        return lines

    def _post_processing(self, conversation: list[dict[str, str]]) -> list[Line]:
        """Converts the guidance program into a list of Lines.
        Cleans up the content for each line."""
        logger.info("Post processing Prolove")
        host = Speaker(name="Zara", gender="female", host=True)
        guest = Speaker(
            name=self.guest.name, gender=self.guest.voice_gender, host=False
        )
        lines = []
        for message in conversation:
            speaker = host if message["role"] == Role.HOST else guest
            content = self._clean_content(message["message"])
            lines.append(Line(speaker=speaker, content=content))
        return lines

    def _clean_content(self, content: str) -> str:
        """Replace placeholder names, remove new lines and extra whitespace."""
        content = content.replace(self.guest.placeholder_name, self.guest.name)
        content = " ".join(content.strip().split())
        return content

    @staticmethod
    def system_message_prompt(identity: str, role: str, mission: str) -> str:
        return (
            SYSTEM_MESSAGE_TEMPLATE.replace("{{identity}}", identity)
            .replace("{{role}}", role)
            .replace("{{mission}}", mission)
        )

    def host_system_message(
        self, mission: str, role: str = HOST_ROLE, identity: str = HOST_IDENTITY
    ) -> str:
        """System message for host"""
        return self.system_message_prompt(identity=identity, role=role, mission=mission)

    def guest_system_message(self, mission: str) -> str:
        """System message for guest"""
        guest_identity = GUEST_IDENTITY.replace("{{name}}", self.guest.placeholder_name)
        guest_identity = guest_identity.replace("{{age}}", str(self.guest.age))
        guest_identity = guest_identity.replace(
            "{{gender}}", self.guest.gender.to_noun()
        )
        guest_identity = guest_identity.replace(
            "{{sexual_orientation}}", self.guest.sexual_orientation
        )
        guest_identity = guest_identity.replace("{{trait}}", self.guest.trait)
        return self.system_message_prompt(
            identity=guest_identity, role=GUEST_ROLE, mission=mission
        )

    def new_top_program(self) -> Program:
        """Creates a new program from top system template"""
        return Program(
            text=TOP_SYSTEM_TEMPLATE,
            llm=self._llm,
            async_mode=True,
        )

    def new_bottom_program(self) -> Program:
        """Creates a new program from bottom system template"""
        return Program(
            text=BOTTOM_SYSTEM_TEMPLATE,
            llm=self._llm,
            async_mode=True,
        )

    @classmethod
    def create(cls, llm: LLM, media_store: MediaStore) -> "RadioShow":
        guest_gender = Gender.random()
        guest_voice_gender = (
            random.choice(GENDERS)
            if guest_gender == Gender.NON_BINARY
            else guest_gender.value
        )
        guest_name = random_name(guest_voice_gender)
        guest_placeholder_name = random_name(guest_voice_gender)
        guest_sexual_orientation = random_sexual_orientation(guest_gender)
        age = random_age()
        topic = random_topic()
        trait = random_trait()
        guest = Guest(
            name=guest_name,
            gender=guest_gender,
            voice_gender=guest_voice_gender,
            sexual_orientation=guest_sexual_orientation,
            age=age,
            trait=trait,
            topic=topic,
            placeholder_name=guest_placeholder_name,
        )
        logger.info("Random guest", guest=guest)

        anecdote = random_anecdote()
        logger.info("Random anecdote", anecdote=anecdote)
        lesson = random_lesson()
        logger.info("Random lesson", lesson=lesson)
        product = random_product()
        logger.info("Random product", product=product)
        k = 2
        host_missions = HostMissions(
            anecdote=anecdote,
            lesson=lesson,
            product=product,
            k=k,
            guest_name=guest.placeholder_name,
        )
        guest_missions = GuestMissions(topic=guest.topic, k=k)
        host_messages = random_host_messages(name=guest.placeholder_name)
        return cls(
            guest=guest,
            host_missions=host_missions,
            guest_missions=guest_missions,
            host_messages=host_messages,
            llm=llm,
            media_store=media_store,
        )


def random_age():
    """Random age for guest.
    Mostly young adults, some middle aged"""
    return int(abs(random.gauss(0.0, 1.0) * 15) + 18)


def guest_message(content: str):
    return {"role": "user", "message": content}


def host_message(content: str):
    return {"role": "assistant", "message": content}
