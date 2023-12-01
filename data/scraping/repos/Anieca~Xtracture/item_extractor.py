import json
from typing import Any

from langchain import PromptTemplate
from langchain.llms import OpenAIChat

from .logger import setup_logger
from .schema import ExtractTarget, Item

logger = setup_logger(__name__)


class GPTItemExtractor:
    TEMPLATE = """Carry out the following instructions.
* Correct any OCR errors in the input text
* Extract the following items according to the given constraints
|Item|Description|Constraint|
{extract_targets}
* Review the extracted results and point out any errors
* If there are any errors, correct them and output the extraction results again
* OUTPUT the extraction results in JSON format, enclosed in triple backticks
  (set the key as string type and the value as array type)

The text is as follows.
{text}
"""

    def __init__(self, model_name: str = "gpt-3.5-turbo") -> None:
        self.gpt = OpenAIChat(model_name=model_name, temperature=0)  # type: ignore
        self.prompt = PromptTemplate(input_variables=["extract_targets", "text"], template=self.TEMPLATE)

    def extract_from_text(self, texts: list[str], targets: list[ExtractTarget]) -> list[Item]:
        extract_targets = "\n".join([f"|{t.key}|{t.description}|{t.constraint}|" for t in targets])

        inputs = self.prompt.format(extract_targets=extract_targets, text="\n".join(texts))
        logger.info(inputs)
        outputs = self.gpt(inputs)
        logger.info(outputs)

        results: dict[str, Any]
        try:
            *_, json_output, _ = outputs.split("```")
            results = json.loads(json_output.replace("json", ""))
        except Exception:
            logger.exception(self.__class__.__name__)
            results = {}
        return [Item(key=target.key, values=results.get(target.key, [])) for target in targets]
