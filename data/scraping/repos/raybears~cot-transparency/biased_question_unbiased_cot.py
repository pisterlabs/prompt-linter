import random
from typing import Sequence

from pydantic import BaseModel
from slist import Slist

from cot_transparency.apis.base import Prompt
from cot_transparency.apis.openai import OpenAIChatPrompt
from cot_transparency.apis.openai.finetune import (
    FinetuneSample,
    join_assistant_preferred_to_completion,
)
from cot_transparency.apis.openai.formatting import (
    append_assistant_preferred_to_last_user,
    append_assistant_preferred_to_next_message,
)
from cot_transparency.data_models.messages import (
    ChatMessage,
    MessageRole,
    StrictChatMessage,
)
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.instructions import (
    BIASED_CONTROL_TOKEN,
    END_SINGLE_SHOT_SEP,
    UNBIASED_CONTROL_TOKEN,
)
from cot_transparency.formatters.interventions.assistant_completion_utils import (
    add_to_final_assistant,
    prepend_to_front_system_message,
)


class BiasedQuestionUnbiasedCOT(BaseModel):
    unbiased_question: Sequence[ChatMessage]
    biased_question: Sequence[ChatMessage]
    # the COT is full_response
    correct_full_response: str
    correct_parsed_response: str
    incorrect_full_response: str
    incorrect_parsed_response: str
    original_biased_task: TaskOutput
    original_unbiased_task: TaskOutput

    @property
    def incorrect_formatter_name(self) -> str:
        return self.original_biased_task.task_spec.formatter_name

    def to_prompt_with_unbiased_response(self) -> Prompt:
        return format_big_brain_question_cot(task=self)

    def to_finetune_sample(self) -> FinetuneSample:
        prompt_messages: Sequence[ChatMessage] = self.biased_question
        new_messages = list(prompt_messages) + [
            ChatMessage(role=MessageRole.assistant, content=self.correct_full_response)
        ]
        # 50% of the time, we put the assistant preferred message as the start of the assistant
        # (so that the assistant learns how to start w/o the instruction)
        # 50% of the time, we put the assistant preferred message as the user's instruction
        # (so that the assistant doesn't forget how to continue)
        seed = self.original_biased_task.task_spec.task_hash
        strict: list[StrictChatMessage] = (
            append_assistant_preferred_to_next_message(prompt=new_messages)
            if random.Random(seed).random() < 0.5
            else append_assistant_preferred_to_last_user(prompt=new_messages)
        )
        return FinetuneSample(messages=strict)

    def to_finetune_sample_using_biased_completion(self) -> FinetuneSample:
        prompt_messages: Sequence[ChatMessage] = self.biased_question
        new_messages = list(prompt_messages) + [
            ChatMessage(role=MessageRole.assistant, content=self.incorrect_full_response)
        ]
        # 50% of the time, we put the assistant preferred message as the start of the assistant
        # (so that the assistant learns how to start w/o the instruction)
        # 50% of the time, we put the assistant preferred message as the user's instruction
        # (so that the assistant doesn't forget how to continue)

        # Option 1
        # Ussr: Question
        # Assistant: Lets think step by step
        # Option 2
        # Ussr: Question, Lets think step by step

        seed = self.original_biased_task.task_spec.task_hash
        strict: list[StrictChatMessage] = (
            append_assistant_preferred_to_next_message(prompt=new_messages)
            if random.Random(seed).random() < 0.5
            else append_assistant_preferred_to_last_user(prompt=new_messages)
        )
        return FinetuneSample(messages=strict)

    def to_finetune_sample_unbiased_context(self) -> FinetuneSample:
        """Converts the biased question to a finetune sample with the unbiased response as the context"""
        prompt_messages: Sequence[ChatMessage] = self.unbiased_question
        new_messages = list(prompt_messages) + [
            ChatMessage(role=MessageRole.assistant, content=self.correct_full_response)
        ]
        seed = self.original_biased_task.task_spec.task_hash
        strict: list[StrictChatMessage] = (
            append_assistant_preferred_to_next_message(prompt=new_messages)
            if random.Random(seed).random() < 0.5
            else append_assistant_preferred_to_last_user(prompt=new_messages)
        )
        return FinetuneSample(messages=strict)

    def to_finetune_sample_control_tokens(self) -> Slist[FinetuneSample]:
        # For the biased response, add NN in the system prompt
        # For the unbiased response, add YY in the system prompt
        prompt_messages = self.biased_question
        added_system_unbiased: Sequence[ChatMessage] = prepend_to_front_system_message(
            messages=prompt_messages, prepend=f"{UNBIASED_CONTROL_TOKEN} "
        )
        added_system_biased: Sequence[ChatMessage] = prepend_to_front_system_message(
            messages=prompt_messages, prepend=f"{BIASED_CONTROL_TOKEN} "
        )
        strict_unbiased = OpenAIChatPrompt(
            messages=join_assistant_preferred_to_completion(
                messages=added_system_unbiased, completion=self.correct_full_response
            )
        )
        strict_biased = OpenAIChatPrompt(
            messages=join_assistant_preferred_to_completion(
                messages=added_system_biased, completion=self.incorrect_full_response
            )
        )
        return Slist(
            [
                FinetuneSample(messages=strict_unbiased.get_strict_messages()),
                FinetuneSample(messages=strict_biased.get_strict_messages()),
            ]
        )


def format_big_brain_question_cot(task: BiasedQuestionUnbiasedCOT) -> Prompt:
    biased_messages: Sequence[ChatMessage] = task.biased_question
    with_correct = add_to_final_assistant(
        biased_messages,
        new_message=" " + task.correct_full_response + END_SINGLE_SHOT_SEP,
    )
    return Prompt(messages=with_correct)
