#!/usr/local/bin/python3
"""Update the AI_CHANGELOG.md file with the latest changes."""
import argparse
from typing import List

from ai_changelog import Commit, get_commits, update_changelog
from ai_changelog.utils import get_prompt, get_llm


def main() -> None:
    """Update the AI_CHANGELOG.md file with the latest changes."""
    parser = argparse.ArgumentParser(
        prog="ai_changelog",
        description="Process command line arguments.",
        epilog="http://github.com/joshuasundance-swca/ai_changelog",
    )
    parser.add_argument(
        "refs",
        type=str,
        default="origin/main^..origin/main",
        help="Reference comparison with standard git syntax",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        help="Model API provider",
        choices=["openai", "anthropic", "anyscale"],
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4",
        help="Model name",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Model temperature",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=500,
        help="Max tokens in output",
    )
    parser.add_argument(
        "--hub_prompt",
        type=str,
        default="joshuasundance/ai_changelog",
        help="Prompt to pull from LangChain Hub",
    )
    parser.add_argument(
        "--context_lines",
        type=int,
        default=5,
        help="Number of context lines for each commit",
    )

    parser.add_argument(
        "--max_concurrency",
        type=int,
        default=0,
        help="Number of concurrent connections to llm provider (0 means no limit)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        help="Run LangChain in verbose mode",
        action="store_true",
    )

    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="Output file",
        default="AI_CHANGELOG.md",
    )

    args = parser.parse_args()

    before_ref, after_ref = args.refs.split("..")

    context_lines = args.context_lines
    provider = args.provider
    llm = args.model
    temperature = args.temperature
    max_tokens = args.max_tokens
    hub_prompt_str = args.hub_prompt
    verbose = args.verbose
    max_concurrency = args.max_concurrency
    output_file = args.output_file

    # Generate the new AI_CHANGELOG.md
    new_commits: List[Commit] = get_commits(
        before_ref=before_ref,
        after_ref=after_ref,
        context_lines=context_lines,
    )

    if new_commits:
        llm = get_llm(provider, llm, temperature, max_tokens)
        prompt = get_prompt(hub_prompt_str)
        update_changelog(
            before_ref=before_ref,
            new_commits=new_commits,
            provider=provider,
            llm=llm,
            prompt=prompt,
            verbose=verbose,
            max_concurrency=max_concurrency,
            output_file=output_file,
        )


if __name__ == "__main__":
    main()
