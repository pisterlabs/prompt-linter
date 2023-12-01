import json
from langchain.prompts import PromptTemplate

from langchain_app.static import SIMPLE_EXTRACTION_TEMPLATE, SIMPLE_FORMAT_INSTRUCTIONS

class ExtractionPromptTemplate:
    def get_runnable(self):
        raise NotImplementedError("Not Implemented.")

class SimpleExtractionPromptTemplate(ExtractionPromptTemplate):
    def __init__(self, schema:dict):
        self.schema = schema
        self.prompt_outline = SIMPLE_EXTRACTION_TEMPLATE
        self.format_outline = SIMPLE_FORMAT_INSTRUCTIONS

    def get_runnable(self):
        format_instructions = self.format_outline.format(schema=json.dumps(self.schema))

        prompt_template = PromptTemplate(
            template=self.prompt_outline,
            input_variables=["context"],
            partial_variables={"format_instructions": format_instructions}
        )

        return prompt_template