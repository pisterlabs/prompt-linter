# ruff: noqa: E501
import random
from pathlib import Path

import structlog
from guidance import Program
from guidance.llms import LLM

from wet_toast_talk_radio.common.dialogue import Line, Speaker, save_lines
from wet_toast_talk_radio.common.log_ctx import show_id_log_ctx
from wet_toast_talk_radio.media_store import MediaStore
from wet_toast_talk_radio.media_store.media_store import ShowId, ShowMetadata, ShowName
from wet_toast_talk_radio.scriptwriter.io import unique_script_filename
from wet_toast_talk_radio.scriptwriter.modern_mindfulness.circumstances import (
    load_circumstances,
)
from wet_toast_talk_radio.scriptwriter.modern_mindfulness.situations import (
    load_situations,
)
from wet_toast_talk_radio.scriptwriter.radio_show import RadioShow

logger = structlog.get_logger()

MEDITATION_TEMPLATE = """{{#system~}}
You are an edgy, satirical writer.
{{~/system}}
{{#block hidden=True}}
{{#user~}}
Your task is to make lists of unlucky yet harmless events that could happen to a person in a given situation.
List the unlucky events one per line. Don't number them or print any other text, just print an unlucky event on each line.
Use a casual tone and informal style.

Here's an example:
Situation:
Taking the metro to go to work.
Events:
Missing your stop.
Getting caught in the door.
Sitting next to a loud chewer.
Accidentally stepping on someone's foot.
Making eye contact with your ex.

Now generate a list of 5 unlucky yet harmless events.
Situation:
{{situation}}.
{{~/user}}
{{#assistant~}}
{{gen 'events' temperature=1.3 max_tokens=500}}
{{~/assistant}}
{{/block}}
{{#system~}}
You are now an enlightened spiritual guru named Orion.
{{~/system}}
{{#user~}}
Your task is to write a guided meditation about the unlucky events and the stressful circumstance listed below.

Situation:
{{situation}}.
Circumstance:
{{circumstance}}
Unlucky events:
{{events}}

This meditation is called "Modern Mindfulness". It is a mindfulness exercise combined with exposure therapy.
Its purpose is for listeners to face and engage with their fears and anxieties in a safe environment.
First, introduce yourself as Orion, a spiritual guru who has transformed anxiety into enlightenment.
Then, Include all events and circumstances in a order that makes chronological sense.
Describe each event in great detail, and focus on their most stressful and frustrating aspects.
You should encourage the listener to remain calm despite the challenges they encounter along the way.
Regularly remind the listener to breathe and relax.
Make the meditation long, descriptive, and in 5000 words.
Finally, end your meditation with your mantra: "Life sucks, but I breathe...".
{{~/user}}
{{#assistant~}}
{{gen 'meditation' temperature=0.6 max_tokens=2500}}
{{~/assistant}}
"""


class ModernMindfulness(RadioShow):
    """Guided meditation radio show combining exposure therapy and mindfulness."""

    def __init__(
        self,
        llm: LLM,
        media_store: MediaStore,
        situation: str | None = None,
        circumstance: str | None = None,
    ):
        self._llm = llm
        self._media_store = media_store
        self._situations = load_situations()
        self.situation = situation if situation else random.choice(self._situations)
        self._circumstances = load_circumstances()
        self.circumstance = (
            circumstance if circumstance else random.choice(self._circumstances)
        )

    @classmethod
    def create(cls, llm: LLM, media_store: MediaStore) -> "ModernMindfulness":
        return cls(llm=llm, media_store=media_store)

    @show_id_log_ctx()
    async def arun(self, show_id: ShowId) -> bool:
        lines = await self.agen()
        self._media_store.put_script_show(show_id=show_id, lines=lines)
        self._media_store.put_script_show_metadata(
            show_id=show_id, metadata=ShowMetadata(ShowName.MODERN_MINDFULNESS)
        )
        return True

    async def awrite(self, output_dir: Path) -> bool:
        lines = await self.agen()
        path = output_dir / unique_script_filename("modern-mindfulness")
        save_lines(path=path, lines=lines)
        return True

    async def agen(self) -> list[Line]:
        logger.info(
            "Async writing modern mindfulness",
            situation=self.situation,
            circumstance=self.circumstance,
        )
        meditation = Program(text=MEDITATION_TEMPLATE, llm=self._llm, async_mode=True)
        written_meditation = await meditation(
            situation=self.situation, circumstance=self.circumstance
        )
        lines = self._post_processing(written_meditation)
        logger.info("Finished writing Modern Mindfulness")
        return lines

    def _post_processing(self, program: Program) -> list[Line]:
        logger.info("Post processing Modern Mindfulness")
        meditation = program["meditation"]
        meditation = " ".join(meditation.strip().split())
        line = Line(
            speaker=Speaker(name="Orion", gender="male", host=True, speaking_rate=1.2),
            content=meditation,
        )
        return [line]
