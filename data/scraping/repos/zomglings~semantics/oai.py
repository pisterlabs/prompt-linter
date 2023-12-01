"""
Semantics extractors that use the Open AI API.
"""

import argparse
import json
import sys

import openai


def handle_models(args: argparse.Namespace) -> None:
    response = openai.Model.list()
    result = response.to_dict_recursive()
    print(json.dumps(result))


def handle_completion(args: argparse.Namespace) -> None:
    with args.prompt as ifp:
        prompt = ifp.read()

    kwargs = {
        k: v
        for k, v in vars(args).items()
        if k not in {"prompt", "func"} and v is not None
    }

    completion = openai.Completion.create(prompt=prompt, **kwargs)
    print(completion)


def generate_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Semantic extraction using OpenAI API")
    parser.set_defaults(func=lambda _: parser.print_help())

    subparsers = parser.add_subparsers()

    models_parser = subparsers.add_parser("models")
    models_parser.set_defaults(func=handle_models)

    completion_parser = subparsers.add_parser("completion")
    completion_parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Model to use for completion. To see all available models, use the following command: 'semantics openai models'. For full documentation: https://beta.openai.com/docs/api-reference/completions/create",
    )
    completion_parser.add_argument(
        "--prompt",
        type=argparse.FileType("r"),
        default=sys.stdin,
        help="File containing prompt for completion.",
    )
    completion_parser.add_argument(
        "--suffix",
        required=False,
        default=None,
        help="Suffix before which insertions should take place in prompt",
    )
    completion_parser.add_argument(
        "--max-tokens",
        required=False,
        type=int,
        default=None,
        help="Number of tokens to generate in completion",
    )
    completion_parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature - how dramatically model strays from highest likelihood token",
    )
    completion_parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Subsample mass - tokens are sampled from those which make up this proportion of the probability mass of most likely tokens",
    )
    completion_parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of completions to generate for prompt",
    )
    completion_parser.add_argument(
        "--logprobs",
        default=None,
        type=int,
        help="Specifies whether to return log probabilities of selected tokens (if --logprobs=0). For higher values of logprobs, also returns log probabilities of that many of the most likely tokens.",
    )
    completion_parser.add_argument(
        "--echo",
        action="store_true",
        help="If this flag is set, the completion will contain the prompt.",
    )
    completion_parser.add_argument(
        "--stop",
        nargs="*",
        default=None,
        help="Up to 4 stop sequences after which the model stops generating tokens.",
    )
    completion_parser.add_argument(
        "--presence-penalty",
        type=float,
        default=None,
        help="Penalty on tokens which are already present in the prompt. Ranges from -2.0 (incentive for the model to generate those tokens) to 2.0 (disincentive).",
    )
    completion_parser.add_argument(
        "--frequency-penalty",
        type=float,
        default=None,
        help="Penalty on tokens which are already present in the completion. Ranges from -2.0 (incentive for the model to generate those tokens) to 2.0 (disincentive).",
    )
    completion_parser.add_argument(
        "--best-of",
        type=int,
        default=None,
        help="Generate this many completions and return the one with highest log probability per token",
    )
    completion_parser.add_argument("--user", default=None, help="Optional session ID")
    completion_parser.set_defaults(func=handle_completion)

    return parser


if __name__ == "__main__":
    parser = generate_cli()
    args = parser.parse_args()
    args.func(args)
