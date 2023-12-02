from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import gpt2anki.data_access.fileio as fileio
from dotenv import load_dotenv
from gpt2anki.data_access.highlight_sources.base import HydratedHighlight
from gpt2anki.domain.prompts_from_string import llmresult_to_qas
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

load_dotenv()
PROMPT_DIR = Path(__file__).parent.parent.parent.parent / "prompts"
assert PROMPT_DIR.exists(), f"Prompts directory does not exist at {PROMPT_DIR}"
SYSTEM_PROMPT = fileio.read_txt(PROMPT_DIR / "martin_prompt.txt")


def initialize_model(model_name: str = "gpt-4") -> ChatOpenAI:
    return ChatOpenAI(model=model_name)


@dataclass(frozen=True)
class HydratedOpenAIPrompt:
    system_message: SystemMessage
    human_message: HumanMessage
    highlight: HydratedHighlight


def highlight_to_msg(highlight: HydratedHighlight) -> HydratedOpenAIPrompt:
    human_message = "<target>{target}</target><context>{context}</context>".format(
        target=highlight.highlight,
        context=highlight.context,
    )
    return HydratedOpenAIPrompt(
        system_message=SystemMessage(content=SYSTEM_PROMPT),
        human_message=HumanMessage(content=human_message),
        highlight=highlight,
    )


@dataclass(frozen=True)
class QAPrompt:
    question: str
    answer: str
    title: str


def finalise_hydrated_questions(
    zipped_outputs: tuple[dict[str, str], HydratedOpenAIPrompt],
) -> QAPrompt:
    match zipped_outputs:
        case (model_outputs, hydrated_prompt):
            return QAPrompt(
                question=model_outputs["question"],
                answer=model_outputs["answer"],
                title=hydrated_prompt.highlight.title,
            )


async def prompts_to_questions(
    hydrated_prompts: list[HydratedOpenAIPrompt],
    model: ChatOpenAI,
) -> list[QAPrompt]:
    prompts = [[x.human_message, x.system_message] for x in hydrated_prompts]

    model_output = await model.agenerate(messages=prompts)
    parsed_outputs = llmresult_to_qas(model_output)

    zipped_outputs = zip(parsed_outputs, hydrated_prompts, strict=True)
    return list(map(finalise_hydrated_questions, zipped_outputs))


async def highlights_to_questions(
    model: ChatOpenAI,
    highlights: Sequence[HydratedHighlight],
) -> list[QAPrompt]:
    hydrated_prompts = [highlight_to_msg(x) for x in highlights]

    questions = await prompts_to_questions(
        hydrated_prompts=hydrated_prompts,
        model=model,
    )

    return questions
