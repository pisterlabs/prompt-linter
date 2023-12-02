import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NamedTuple

import openai
from tqdm import tqdm

from common import (
    calculate_cost,
    get_key,
    get_result,
    init_argparser,
    log_args,
    logger,
    make_chat_request,
    make_msg,
)
from metrics import (
    MetricPrediction,
    MetricReference,
    StructureFormat,
    calculate_metrics,
)

USER_PROMPTS = [
    # 0
    "What are the causes, effects and relations in the following text?",
    # 1
    """\
What are the causes, effects and relation in the following text?
The relation must be one of "cause", "enable", or "prevent".
The causes and effects must be spans of the text. There is only one relation.

The response should be formatted as this:
Cause: <text>
Effect: <text>
Relation: <text>

When there are multiple causes or effects, separate them by " | ". Don't add quotes around the extractions.
""",
    # 2
    """\
What are the causes, effects and relation in the following text?
The relation must be one of "cause", "enable", or "prevent".
The causes and effects must be spans of the text. There is only one relation.

The response should be formatted as this:
`[Cause] <cause text> [Relation] <relation> [Effect] <effect text>`

When there are multiple causes or effects, separate them by " | ". Don't add quotes around the extractions.
""",
]
SYSTEM_PROMPTS = [
    # 0
    "You are a helpful assistant that extract causes, effects, and"
    " relations from text. The format is `[Cause] <cause text> [Relation]"
    " <relation> [Effect] <effect text>`.",
    # 1
    "You are a helpful assistant that extract causes, effects, and a relation from"
    " text.",
]


def make_extraction_request(
    model: str,
    text: str,
    examples: list[dict[str, str]],
    user_prompt: str,
    system_prompt: str,
) -> dict[str, Any]:
    return make_chat_request(
        model=model,
        messages=generate_extraction_messages(
            text, examples, user_prompt, system_prompt
        ),
        temperature=0,
        seed=0,
    )


def gen_extraction_example_exchange(
    examples: list[dict[str, str]], prompt: str
) -> Iterator[dict[str, str]]:
    prompt_msg = make_msg("user", prompt)
    for example in examples:
        yield prompt_msg
        yield make_msg("user", example["context"])
        yield make_msg("assistant", example["answers"])


def generate_extraction_messages(
    text: str,
    examples: list[dict[str, str]],
    user_prompt: str,
    system_prompt: str,
) -> list[dict[str, str]]:
    return [
        make_msg(
            "system",
            system_prompt,
        ),
        *gen_extraction_example_exchange(examples, user_prompt),
        make_msg("user", user_prompt),
        make_msg("user", text),
    ]


class ExtractionResult(NamedTuple):
    responses: list[dict[str, Any]]
    predictions: list[MetricPrediction]
    metrics: dict[str, float]


def extract_clauses(
    model: str,
    demonstration_examples: list[dict[str, str]],
    inputs: list[MetricReference],
    extraction_mode: StructureFormat,
    user_prompt: str,
    system_prompt: str,
) -> ExtractionResult:
    responses: list[dict[str, Any]] = []
    predictions: list[MetricPrediction] = []

    for example in tqdm(inputs):
        response = make_extraction_request(
            model,
            example["context"],
            demonstration_examples,
            user_prompt,
            system_prompt,
        )
        pred: MetricPrediction = {
            "id": example["id"],
            "prediction_text": get_result(response),
        }
        responses.append(response)
        predictions.append(pred)

    metrics = calculate_metrics(predictions, inputs, extraction_mode)

    return ExtractionResult(responses, predictions, metrics)


def run_extraction(
    model: str,
    demonstration_examples_path: Path | None,
    input_path: Path,
    output_path: Path | None,
    metric_path: Path | None,
    extraction_mode: StructureFormat,
    user_prompt: str,
    system_prompt: str,
) -> None:
    if demonstration_examples_path is not None:
        demonstration_examples = json.loads(demonstration_examples_path.read_text())
    else:
        demonstration_examples = []

    examples: list[MetricReference] = json.loads(input_path.read_text())["data"]

    responses, predictions, metrics = extract_clauses(
        model=model,
        demonstration_examples=demonstration_examples,
        inputs=examples,
        extraction_mode=extraction_mode,
        user_prompt=user_prompt,
        system_prompt=system_prompt,
    )

    output = [
        {
            "id": example["id"],
            "text": example["context"],
            "pred": pred["prediction_text"],
            "answer": example["answers"],
        }
        for example, pred in zip(examples, predictions)
    ]
    if output_path is not None:
        output_path.write_text(json.dumps(output, indent=2))
    else:
        print(json.dumps(output, indent=2))

    metrics["cost"] = sum(calculate_cost(model, response) for response in responses)
    print("\nMetrics:")
    print(json.dumps(metrics, indent=2))
    if metric_path is not None:
        metric_path.write_text(json.dumps(metrics, indent=2))


def main() -> None:
    parser = init_argparser()
    parser.add_argument(
        "--examples",
        type=Path,
        help="The path to file with the demonstration examples.",
    )
    parser.add_argument(
        "--mode",
        default="tags",
        choices=["tags", "lines"],
        type=StructureFormat,
        help="The format of the structured output.",
    )
    args = parser.parse_args()
    log_args(args, path=args.args_path)

    logger.config(args.log_file, args.print_logs)
    openai.api_key = get_key(args.key_file, args.key_name)

    if args.prompt < 0 or args.prompt >= len(USER_PROMPTS):
        raise IndexError(f"Invalid user prompt index: {args.prompt}")

    if args.sys_prompt < 0 or args.sys_prompt >= len(USER_PROMPTS):
        raise IndexError(f"Invalid system prompt index: {args.prompt}")

    run_extraction(
        model=args.model,
        demonstration_examples_path=args.examples,
        input_path=args.input,
        output_path=args.output,
        metric_path=args.metrics_path,
        extraction_mode=args.mode,
        user_prompt=USER_PROMPTS[args.prompt],
        system_prompt=SYSTEM_PROMPTS[args.sys_prompt],
    )


if __name__ == "__main__":
    main()
