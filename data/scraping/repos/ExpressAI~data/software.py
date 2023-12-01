from __future__ import annotations

import openai

from promptware.constants import (
    ApplicationCategory,
    ApplicationSubcategory,
    DesignPatternType,
    LanguageType,
    LicenseType,
    PlatformType,
    TaskType,
)
from promptware.info import SoftwareInfo
from promptware.kernels.plm import PLMKernelConfig
from promptware.promptware import PromptConfig, Promptware


class InstructionTypePromptware(Promptware):
    def _info(self) -> SoftwareInfo:
        return SoftwareInfo(
            description="This promptware is used to identify the sentiment of a "
            "sentence (positive or negative) based on some learning",
            creator="Promptware Authors",
            homepage="https://github.com/expressai/promptware",
            reference="",
            codebase_url="https://github.com/expressai/promptware/tree/main/softwares",
            license=LicenseType.apache_2_0,
            research_tasks=[TaskType.text_classification],
            application_categories=[ApplicationCategory.classification],
            application_subcategories=[ApplicationSubcategory.sentiment_analysis],
            original_platform=PlatformType.gpt3,
            design_pattern=DesignPatternType.standalone,
            source_language=LanguageType.en,
            target_language=LanguageType.en,
        )

    def _kernel_configs(self):
        return {
            "openai": PLMKernelConfig(
                platform="openai",
                model_name="gpt-3.5-turbo",
                max_tokens=256,
                temperature=0.7,
            )
        }

    def _software_configs(self):
        return {
            "extract_verb_object": PromptConfig(
                name="extract_verb_object",
                description="extract_verb_object",
                instruction="Extract the verb-object structure phrase which is no"
                " more than 10 words from this sentence and the first word is a verb"
                " with a capital letter:",
                demonstration=[],
                prompt_template=lambda input: f"{input['text']}",
                task=TaskType.text_classification,
            ),
            "condense": PromptConfig(
                name="condense",
                description="condense",
                instruction="Given a question asked by user, sum up user's intention"
                " starting with a verb using less than 10 words:",
                demonstration=[],
                prompt_template=lambda input: f"{input['text']}",
                task=TaskType.text_classification,
            ),
        }

    def _example(self):
        return {"input": {"text": "I love this movie."}, "output": "positive"}

    def execute(self, input):
        if "?" in input["text"]:
            code = self.get_code(input, self.software_configs["condense"])
        else:
            code = self.get_code(input, self.software_configs["extract_verb_object"])

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": code}]
        )

        output = completion["choices"][0]["message"]["content"]

        result = self.normalize_output(output)

        return result
