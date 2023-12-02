import json
from pathlib import Path
from typing import Optional, Tuple

import click
import openai
import pathspec
import tiktoken


def estimate_cost(prompt: str) -> Tuple[int, float]:
    encoder = tiktoken.encoding_for_model("gpt-4")
    num_tokens = len(encoder.encode(prompt))
    cost_per_token = 0.004 / 1000  # Conservative estimate
    cost = num_tokens * cost_per_token
    return num_tokens, cost


def complete(prompt: str, model: str) -> str:
    result = []
    for chunk in openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        stream=True,
    ):
        content = chunk.choices[0].get("delta", {}).get("content")  # type: ignore
        if content is not None:
            result.append(content)
            click.echo(content, nl=False)
    return "".join(result)


def summarize_text(
    text: str, path: Path, model: str, repo_name: Optional[str] = None
) -> str:
    language = None
    try:
        language = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".md": "markdown",
            ".ini": "ini",
            ".toml": "toml",
            ".yaml": "yaml",
            ".json": "json",
            ".cfg": "cfg",
        }[path.suffix]
    except KeyError:
        pass

    prompt = (
        f"Your task is to summarize files.\n"
        f"- These files belongs to the code repo: {repo_name}.\n"
        if repo_name
        else ""
        f"- Do not start the summary with 'This file contains [summary]'. Rather start with '[summary]'.\n"
        f"- Minimize prose. For example, do not say: 'The setup.py file contains information about the package and its dependencies.' "
        f"Instead, say: 'The package depends on the following packages: numpy, scipy, and pandas.'\n"
        f"- Include code examples of how to use the public classes and functions defined in a file. "
        f"Do not discuss private classes or functions. In Python, private objects are marked with a leading underscore in their name, e.g. `_func()`\n"
        f"- Write in a style combining that of Simon Willison and Jeremy Howard\n"
        f"- Use mermaid diagrams to explain the structure of complex data models or processes\n"
        f"\n"
        f"## File: {path}\n\n"
        f"### Contents\n\n```"
    )
    if language:
        prompt += language
    prompt += f"\n{text}\n```\n\n### Summary\n\n"

    num_tokens, cost = estimate_cost(prompt)
    click.echo(f"Number of tokens: {num_tokens}", err=True)
    click.echo(f"Estimated cost: ${cost}", err=True)

    # Get confirmation from the user
    if not click.confirm("Continue?", err=True):
        raise click.Abort()

    summary = complete(prompt, model)
    return summary


@click.group()
def cli():
    pass


@cli.command()
@click.argument("repo_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--model",
    default="gpt-4",
    type=click.Choice(["gpt-4", "gpt-3.5-turbo"]),
    help="Which model to use for generation.",
)
def readme(repo_dir, model):
    """
    Generate README.md content for the repo at REPO_DIR.
    """
    repo_dir = Path(repo_dir)

    # Ask GPT4 to summarize each document
    summaries_path = repo_dir / "summaries.json"
    documents = json.loads(summaries_path.read_text())

    # Create a prompt
    summaries = "\n\n".join(
        f"## {document['path']}\n\n{document['summary']}" for document in documents
    )

    prompt = (
        f"Create a README.md file for the repository: {repo_dir.absolute().name}. "
        f"The repository contains the following files:\n"
        f"{summaries}\n\n"
        f"## README.md\n\n"
    )

    # Estimate cost
    # Note that this excludes a few tokens e.g. to distinguish messages
    encoder = tiktoken.encoding_for_model("gpt-4")
    num_tokens = len(encoder.encode(prompt))
    cost_per_token = 0.004 / 1000  # Conservative estimate
    cost = num_tokens * cost_per_token

    click.echo(f"Number of tokens: {num_tokens}", err=True)
    click.echo(f"Estimated cost: ${cost:.2f}", err=True)

    # Get confirmation from the user
    if not click.confirm("Continue?", err=True):
        return

    # Generate README.md
    complete(prompt, model)


@cli.command()
@click.argument("path_or_dir", type=click.Path(exists=True))
@click.option(
    "--model",
    default="gpt-4",
    type=click.Choice(["gpt-4", "gpt-3.5-turbo"]),
    help="Which model to use for generation.",
)
def summarize(path_or_dir, model):
    """
    Summarize a file or directory.
    """
    path_or_dir = Path(path_or_dir)

    if path_or_dir.is_file():
        text = path_or_dir.read_text()
        summary = summarize_text(text, path=path_or_dir, model=model)
    elif path_or_dir.is_dir():
        # Traverse up to a git directory
        repo_dir = path_or_dir
        while not (repo_dir / ".git").exists():
            repo_dir = repo_dir.parent
            if repo_dir == repo_dir.root:
                repo_dir = None
                break

        # Read the .gitignore file if present
        spec = None
        if repo_dir:
            gitignore_path = repo_dir / ".gitignore"
            if gitignore_path.exists():
                spec = pathspec.PathSpec.from_lines(
                    "gitwildmatch", gitignore_path.read_text().splitlines()
                )

        # Traverse through the directory tree and collect text, respecting the .gitignore if present
        documents = []
        # TODO: Does adding filenames improve the quality of the generated README?
        for path in path_or_dir.rglob("*"):
            if not path.is_file():
                continue
            if spec and spec.match_file(path):
                continue
            if path.parts[0] == ".git":
                continue
            if path.name == ".gitignore":
                continue

            try:
                text = path.read_text()
            except UnicodeDecodeError:
                pass
            else:
                documents.append({"path": path, "text": text})

        click.echo(f"Number of documents: {len(documents)}", err=True)

        # Ask GPT4 to summarize each document
        summaries_path = path_or_dir / "summaries.json"
        if summaries_path.exists():
            documents = json.loads(summaries_path.read_text())
        else:
            repo_name = repo_dir and repo_dir.name
            for document in documents:
                click.echo(f"\n\n## {document['path']}\n", err=True)
                try:
                    text = document["path"].read_text()
                except UnicodeDecodeError:
                    continue
                summary = summarize_text(
                    text, path=path_or_dir, repo_name=repo_name, model=model
                )
                document["summary"] = summary

            def get_saved_document(document):
                return {
                    "path": str(document["path"]),
                    "summary": document["summary"],
                }

            saved_documents = [get_saved_document(document) for document in documents]
            summaries_path.write_text(json.dumps(saved_documents, indent=2))
    else:
        raise click.ClickException(f"Path is not a file or directory: {path_or_dir}")


if __name__ == "__main__":
    cli()
