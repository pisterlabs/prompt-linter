from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from web_migration_assistant.output_utils import extract_code_snippets
from langchain.prompts import PromptTemplate
from typing import Tuple
from web_migration_assistant.vue_to_react_templates import (
    INJECT_FRAMEWORK_FROM,
    INJECT_FRAMEWORK_TO,
    INJECT_LANGUAGE,
    INJECT_LANGUAGE_STORY,
    INJECT_OUTPUT_STRUCTURE_TEMPLATE,
    INJECT_OUTPUT_STRUCTURE_TEMPLATE_STORY,
    TEMPLATE_V1,
    TEMPLATE_STORY_V1
)


def convert(inject_code_from: str) -> Tuple[str, str]:
    llm = ChatOpenAI(model="gpt-4", temperature=0.9)

    prompt = PromptTemplate(
        input_variables=[
            "language",
            "code_from",
            "framework_from",
            "framework_to",
            "output_structure_template",
        ],
        template=TEMPLATE_V1,
    )
    llm_input = prompt.format(
        language=INJECT_LANGUAGE,
        framework_from=INJECT_FRAMEWORK_FROM,
        framework_to=INJECT_FRAMEWORK_TO,
        code_from=inject_code_from,
        output_structure_template=INJECT_OUTPUT_STRUCTURE_TEMPLATE
        # test="test"
    )

    llm_output = llm.predict(llm_input)

    try:
        code_new = extract_code_snippets(llm_output)
    except IndexError:
        print("IndexError")
        code_new = llm_output


    prompt_story = PromptTemplate(
        input_variables=[
            "language",
            "code_new",
            "framework_to",
            "output_structure_template_story",
        ],
        template=TEMPLATE_STORY_V1,
    )
    llm_input = prompt_story.format(
        language=INJECT_LANGUAGE_STORY,
        framework_to=INJECT_FRAMEWORK_TO,
        code_new=code_new,
        output_structure_template_story=INJECT_OUTPUT_STRUCTURE_TEMPLATE_STORY
        # test="test"
    )

    llm_output = llm.predict(llm_input)
    try:
        code_story = extract_code_snippets(llm_output)
    except IndexError:
        print("IndexError")
        code_story = llm_output

    return code_new, code_story
