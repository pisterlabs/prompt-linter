import os
import functools
from dataclasses import dataclass
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from pathlib import Path
import any_serde

import yaml


@functools.lru_cache
def get_claude_key() -> str:
    return os.environ["CLAUDE_KEY"]


@dataclass
class Turn:
    user_message: str
    assistant_message: str | None


@dataclass
class Trajectory:
    # system_message
    turns: list[Turn]


@dataclass
class GenerationResult:
    inference_trajectory: Trajectory
    generation_result: str


def generate(
    trajectory: Trajectory,
    max_tokens_to_sample: int = 500,
    # TODO: inference params
    save_path: Path | None = None,
) -> str:
    print(f"{save_path=}")
    if save_path is not None and save_path.exists():
        with save_path.open("rt") as fin_result:
            generation_result_data = yaml.load(fin_result, Loader=yaml.SafeLoader)
        generation_result = any_serde.from_data(GenerationResult, generation_result_data)
        if generation_result.inference_trajectory == trajectory:
            return generation_result.generation_result

    anthropic = Anthropic(
        api_key=get_claude_key(),
    )

    prompt = ""
    for turn_idx, turn in enumerate(trajectory.turns):
        prompt += f"{HUMAN_PROMPT} {turn.user_message}"
        if turn.assistant_message is not None:
            if turn_idx == len(trajectory.turns) - 1:
                raise ValueError("Cannot generate human messages!")
            prompt += f"{AI_PROMPT} {turn.assistant_message}"
        else:
            assert turn_idx == len(trajectory.turns) - 1
            prompt += f"{AI_PROMPT}"

    completion = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=max_tokens_to_sample,
        prompt=prompt,
    )

    response = completion.completion.lstrip()

    if save_path is not None:
        generation_result = GenerationResult(
            inference_trajectory=trajectory,
            generation_result=response,
        )
        generation_result_data = any_serde.to_data(GenerationResult, generation_result)
        print(f"saving to {save_path}")
        with save_path.open("wt") as fout_result:
            yaml.dump(generation_result_data, fout_result)

    return response
