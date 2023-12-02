# noqa
from __future__ import annotations

import openai

import promptware
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

INSTRUCTION = (
    "Given a text, select the most appropriate category from the "
    "list below that characterizes its task type without additional"
    " words. If the text involves  programming languages, "
    "such as python, SQL, c++, java, the selected category "
    'should contain "code":'
)

categories = [
    "text writing based on a prompt",
    "explain the reason",
    "compare different objects",
    "specialized educational dialogs",
    "open-ended conversation",
    "text classification",
    "ordering",
    "sentiment analysis",
    "code language classification",
    "code generation",
    "code implementation",
    "data generation",
    "advice-giving",
    "information enumeration",
    "recommendation",
    "how to question",
    "text rewriting",
    "code rewriting",
    "information extraction",
    "text summarization",
    "code explanation",
    "text explanation",
    "translation",
    "value judgment",
    "hack identification",
    "Internet search",
    "new information",
]

PROMPT = INSTRUCTION + "\n" + "\n".join([f"- {c}" for c in categories])

print(PROMPT)


class InstructionTaskPromptware(Promptware):
    def _info(self) -> SoftwareInfo:
        return SoftwareInfo(
            description="This promptware is used to identify the task type of "
            "an instruction.",
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
                max_tokens=64,
                temperature=1,
            )
        }

    def _software_configs(self):
        return {
            "extract_instruction_task": PromptConfig(
                name="extract_instruction_task",
                description="extract_instruction_task",
                instruction=PROMPT,
                demonstration=[],
                prompt_template=lambda input: f"\nText: {input['text']}",
                task=TaskType.text_classification,
            ),
        }

    def _example(self):
        return {"input": {"text": "I love this movie."}, "output": "positive"}

    def postprocess(self, text: str) -> str:
        if ":" in text:
            return text.split(":")[1].replace(".", "").strip().lower()
        else:
            return text.replace(".", "").strip().lower()

    def execute(self, input):
        openai.api_key = promptware.os_api_key

        code = self.get_code(input, self.software_configs["extract_instruction_task"])

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": code}]
        )
        output = completion["choices"][0]["message"]["content"]

        result = self.postprocess(self.normalize_output(output))

        return result
