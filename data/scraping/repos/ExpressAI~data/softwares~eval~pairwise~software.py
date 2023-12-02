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

# - The response should strictly meet the user's description of the scenario and contain valuable information for the user
# - the response should be informative, visual, logical, and actionable.
# - the response should avoid being vague, controversial, or off-topic.
# - the response should not refuse to answer the question or instruction.


PROMPT = """
You're given a question or an instruction from a user and two responses; Please evaluate the quality of the responses based on the following criteria to see which one is better and explain in detail. 
Start with "Response 1", "Response 2":

- Relevance: Check whether the response directly addresses the question or topic at hand. It should provide pertinent information and stay on track, without deviating from the subject matter.
- Coherence: Assess the flow and organization of the response. The ideas should be connected logically and should make sense within the context of the question or discussion.
- Accuracy: Verify the factual correctness of the information provided. Cross-check with reliable sources to ensure that the AI has not produced misleading or outdated information.
- Completeness: Determine if the response provides a comprehensive answer to the question or topic, covering all relevant aspects and details.
- Clarity: Evaluate the response for simplicity and understandability. The language should be clear and concise, avoiding any unnecessary jargon or complexity.
- Grammar and Syntax: Examine the response for proper grammar, punctuation, and syntax. The response should be well-written and free of errors.
- Appropriateness: Assess the tone and language used in the response. The AI should maintain a neutral and respectful tone, avoiding any offensive or inappropriate content.
"""

print(PROMPT)
print("---------------\n")

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
            "eval/pairwise": PromptConfig(
                name="eval/pairwise",
                description="pairwise evaluation",
                instruction=PROMPT,
                demonstration=[],
                prompt_template=lambda input: f"\nQuestion: {input['question']}\nResponse 1: {input['response1']}\nResponse 2: {input['response2']}\n",
                task=TaskType.text_classification,
            ),
        }

    def _example(self):
        return {"input": {"question": "I love this movie.", "response1":"", "response2":""}, "output": "positive"}

    def postprocess(self, text: str) -> str:
        return text.replace(".", "").strip().lower()

    def execute(self, input):


        openai.api_key = promptware.os_api_key

        code = self.get_code(input, self.software_configs["eval/pairwise"])

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": code}]
        )
        output = completion["choices"][0]["message"]["content"]

        result = self.normalize_output(output)

        return result
