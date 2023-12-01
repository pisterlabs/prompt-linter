from typing import Optional, Sequence
from collections import defaultdict

from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate

from athena import emit_meta
from athena.programming import Exercise, Submission

from module_programming_llm.config import BasicApproachConfig
from module_programming_llm.helpers.llm_utils import (
    get_chat_prompt_with_formatting_instructions, 
    num_tokens_from_string, 
    num_tokens_from_prompt, 
    predict_and_parse
)
from module_programming_llm.helpers.utils import format_grading_instructions, get_diff


class FileGradingInstruction(BaseModel):
    file_name: str = Field(description="File name")
    grading_instructions: str = Field(description="Grading instructions relevant for this file")


class SplitGradingInstructions(BaseModel):
    """Collection of grading instructions split by file"""
    items: Sequence[FileGradingInstruction] = Field(description="File grading instructions")


# pylint: disable=too-many-locals
async def split_grading_instructions_by_file(
        exercise: Exercise, 
        submission: Submission,
        prompt: ChatPromptTemplate,
        config: BasicApproachConfig, 
        debug: bool
    ) -> Optional[SplitGradingInstructions]:
    """Split the general grading instructions by file

    Args:
        exercise (Exercise): Exercise to split the grading instructions for (respecting the changed files)
        submission (Submission): Submission to split the grading instructions for (respecting the changed files)
        prompt (ChatPromptTemplate): Prompt template to check for grading_instructions
        config (BasicApproachConfig): Configuration

    Returns:
        Optional[SplitGradingInstructions]: Split grading instructions, None if it is too short or too long
    """

    grading_instructions = format_grading_instructions(exercise.grading_instructions, exercise.grading_criteria)

    # Return None if the grading instructions are too short
    if (grading_instructions is None 
            or num_tokens_from_string(grading_instructions) <= config.split_grading_instructions_by_file_prompt.tokens_before_split):
        return None

    # Return None if the grading instructions are not in the prompt
    if "grading_instructions" not in prompt.input_variables:
        return None
    
    model = config.model.get_model()  # type: ignore[attr-defined]
    
    template_repo = exercise.get_template_repository()
    solution_repo = exercise.get_solution_repository()
    submission_repo = submission.get_repository()

    changed_files_from_template_to_solution = get_diff(
        src_repo=template_repo, 
        dst_repo=solution_repo, 
        file_path=None, 
        name_only=True
    ).split("\n")

    changed_files_from_template_to_submission = get_diff(
        src_repo=template_repo, 
        dst_repo=submission_repo, 
        file_path=None, 
        name_only=True
    ).split("\n")

    chat_prompt = get_chat_prompt_with_formatting_instructions(
        model=model, 
        system_message=config.split_grading_instructions_by_file_prompt.system_message, 
        human_message=config.split_grading_instructions_by_file_prompt.human_message, 
        pydantic_object=SplitGradingInstructions
    )

    prompt_input = {
        "grading_instructions": grading_instructions,
        "changed_files_from_template_to_solution": ", ".join(changed_files_from_template_to_solution),
        "changed_files_from_template_to_submission": ", ".join(changed_files_from_template_to_submission)
    }

    # Return None if the prompt is too long
    if num_tokens_from_prompt(chat_prompt, prompt_input) > config.max_input_tokens:
        return None

    split_grading_instructions = await predict_and_parse(
        model=model, 
        chat_prompt=chat_prompt, 
        prompt_input=prompt_input, 
        pydantic_object=SplitGradingInstructions,
        tags=[
            f"exercise-{exercise.id}",
            f"submission-{submission.id}",
            "split-grading-instructions-by-file"
        ]
    )

    if debug:
        emit_meta("file_grading_instructions", {
            "prompt": chat_prompt.format(**prompt_input),
            "result": split_grading_instructions.dict() if split_grading_instructions is not None else None
        })

    if split_grading_instructions is None or not split_grading_instructions.items:
        return None

    # Join duplicate file names (some responses contain multiple grading instructions for the same file)
    file_grading_instructions_by_file_name = defaultdict(list)
    for file_grading_instruction in split_grading_instructions.items:
        file_grading_instructions_by_file_name[file_grading_instruction.file_name].append(file_grading_instruction)

    split_grading_instructions.items = [
        FileGradingInstruction(
            file_name=file_name,
            grading_instructions="\n".join(
                file_grading_instruction.grading_instructions
                for file_grading_instruction in file_grading_instructions
            )
        )
        for file_name, file_grading_instructions in file_grading_instructions_by_file_name.items()
    ]

    return split_grading_instructions
