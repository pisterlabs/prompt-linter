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


class ChatGPTPromptware(Promptware):
    def _info(self) -> SoftwareInfo:
        return SoftwareInfo(
            description="Response generation by ChatGPT",
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
                max_tokens=512,
                temperature=1,
            )
        }

    def _software_configs(self):
        return {
            "chatting": PromptConfig(
                name="chatting",
                description="chatting",
                instruction="",
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

        code = self.get_code(input, self.software_configs["chatting"])

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. "
                    "Answer the following question in detail.",
                },
                {"role": "user", "content": code},
            ],
        )
        output = completion["choices"][0]["message"]["content"]

        result = self.normalize_output(output)

        return result
