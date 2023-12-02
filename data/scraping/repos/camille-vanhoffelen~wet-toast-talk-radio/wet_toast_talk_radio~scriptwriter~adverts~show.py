from dataclasses import dataclass
from pathlib import Path

import structlog
from guidance import Program
from guidance.llms import LLM

from wet_toast_talk_radio.common.dialogue import Line, Speaker, save_lines
from wet_toast_talk_radio.common.log_ctx import show_id_log_ctx
from wet_toast_talk_radio.media_store import MediaStore
from wet_toast_talk_radio.media_store.media_store import ShowId, ShowMetadata, ShowName
from wet_toast_talk_radio.scriptwriter.adverts.products import random_product
from wet_toast_talk_radio.scriptwriter.adverts.strategies import random_strategies
from wet_toast_talk_radio.scriptwriter.io import unique_script_filename
from wet_toast_talk_radio.scriptwriter.radio_show import RadioShow

logger = structlog.get_logger()

PRODUCT_TEMPLATE = """{{#system~}}
You are good at product marketing.
{{~/system}}
{{#user~}}
Your task is to think of a name for the following product:
{{description}}
Just write the name, do not print any other text.
{{~/user}}
{{#assistant~}}
{{gen 'name' temperature=1.2 max_tokens=10}}
{{~/assistant}}
{{#user~}}
Now generate the name of the company that sells this product.
Just write the name of the company, do not print any other text.
{{~/user}}
{{#assistant~}}
{{gen 'company' temperature=1.2 max_tokens=10}}
{{~/assistant}}"""

ADVERT_TEMPLATE = """{{#system~}}
You are good at product marketing.
You write in a casual and informal manner.
{{~/system}}
{{#user~}}
Your task is write the text for a good endorsement ad. The product is as follows:

Product name: {{name}}
Product description: {{description}}
Company name: {{company}}

Here is the advertising strategy you must follow.

1. Make a catchy introduction
{{#each strategies~}}
{{@index + 2}}. {{this}}
{{/each}}

Follow each step. Be detailed and specific.
You cannot include sound effects, soundbites or music.
Start with "And now for a word from our sponsors. " and end with "Buy {{name}} today!".
{{~/user}}
{{#assistant~}}
{{gen 'advert' temperature=0.7 max_tokens=800}}
{{~/assistant}}"""


@dataclass
class Product:
    name: str
    description: str
    company: str


class Advert(RadioShow):
    """A radio show where the host reads out an advert for an absurd product."""

    def __init__(
        self,
        product_description: str,
        strategies: list[str],
        llm: LLM,
        media_store: MediaStore,
    ):
        self._llm = llm
        self._media_store = media_store
        self.product_description = product_description
        self.strategies = strategies

    @classmethod
    def create(cls, llm: LLM, media_store: MediaStore) -> "Advert":
        product_description = random_product()
        logger.info("Random product description", description=product_description)
        strategies = random_strategies(k_part_1=4, k_part_2=3)
        logger.info("Random strategies", strategies=strategies)
        return cls(
            product_description=product_description,
            strategies=strategies,
            llm=llm,
            media_store=media_store,
        )

    @show_id_log_ctx()
    async def arun(self, show_id: ShowId) -> bool:
        lines = await self.agen()
        self._media_store.put_script_show(show_id=show_id, lines=lines)
        self._media_store.put_script_show_metadata(
            show_id=show_id, metadata=ShowMetadata(ShowName.ADVERTS)
        )
        return True

    async def awrite(self, output_dir: Path) -> bool:
        lines = await self.agen()
        path = output_dir / unique_script_filename("advert")
        save_lines(path=path, lines=lines)
        return True

    async def agen(self) -> list[Line]:
        logger.info("Async writing Advert")

        product = Program(text=PRODUCT_TEMPLATE, llm=self._llm, async_mode=True)
        product = await product(description=self.product_description)

        logger.info(
            "Generated product", name=product["name"], company=product["company"]
        )
        advert = Program(text=ADVERT_TEMPLATE, llm=self._llm, async_mode=True)
        advert = await advert(
            name=product["name"],
            description=self.product_description,
            company=product["company"],
            strategies=self.strategies,
        )
        logger.info("Finished writing Advert")

        lines = self._post_processing(program=advert)
        return lines

    def _post_processing(self, program: Program) -> list[Line]:
        logger.info("Post processing Advert")
        content = program["advert"]
        # For discounts
        content = content.replace("%", " percent")
        content = " ".join(content.strip().split())
        line = Line(
            speaker=Speaker(name="Ian", gender="male", host=True), content=content
        )
        return [line]
