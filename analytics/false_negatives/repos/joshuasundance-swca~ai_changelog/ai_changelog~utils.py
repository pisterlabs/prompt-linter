"""Utility functions for the ai_changelog package"""
import os
import subprocess
from typing import Any, List, Union

from langchain import hub
from langchain.chains.base import Chain
from langchain.chains.openai_functions import (
    create_structured_output_chain,
)
from langchain.chat_models import ChatOpenAI, ChatAnyscale, ChatAnthropic
from langchain.chat_models.base import BaseChatModel
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableConfig

from ai_changelog.pydantic_models import CommitDescription, CommitInfo, Commit
from ai_changelog.string_templates import hum_msg, sys_msg


def get_llm(
    provider: str,
    model: str,
    temperature: float = 0.5,
    max_tokens: int = 1000,
) -> BaseChatModel:
    provider_model_dict = {
        "openai": ChatOpenAI,
        "anthropic": ChatAnthropic,
        "anyscale": ChatAnyscale,
    }
    try:
        model_class = provider_model_dict[provider]
    except KeyError as e:
        raise ValueError(f"Unknown provider {provider}") from e
    return model_class(model=model, temperature=temperature, max_tokens=max_tokens)


def get_prompt(
    hub_prompt_str: str = "joshuasundance/ai_changelog",
) -> ChatPromptTemplate:
    return (
        ChatPromptTemplate.from_messages(
            [
                ("system", sys_msg),
                ("human", hum_msg),
                ("human", "Tip: Make sure to answer in the correct format"),
            ],
        )
        if hub_prompt_str == "joshuasundance/ai_changelog"
        else hub.pull(hub_prompt_str)
    )


def get_non_openai_chain(llm: BaseChatModel) -> Chain:
    codellama_prompt_template = hub.pull("joshuasundance/ai_changelog_codellama")
    parser = PydanticOutputParser(pydantic_object=CommitDescription)
    fixing_parser = OutputFixingParser.from_llm(
        parser=parser,
        llm=llm
        if not isinstance(llm, ChatAnyscale)
        else ChatAnyscale(model_name="meta-llama/Llama-2-7b-chat-hf", temperature=0),
    )
    return codellama_prompt_template | llm | fixing_parser


def get_timestamp(commit_hash: str, format_str: str = "%cD") -> str:
    """Get the timestamp for a commit hash"""
    cmd = ["git", "show", "-s", f"--format={format_str}", commit_hash]
    return subprocess.check_output(cmd).decode().strip()


def rev_parse(ref: str) -> str:
    """Get the commit hash for a reference"""
    return subprocess.check_output(["git", "rev-parse", ref]).decode().strip()


def dt_diffs_from_hashes(
    hashes: List[str],
    context_lines: int = 5,
) -> List[List[str]]:
    cmd = "git --no-pager show --no-notes {commit} -s --pretty=%cd --quiet --patch -U{context_lines}"
    return [
        output.split("\n", maxsplit=1)
        for output in [
            subprocess.check_output(
                cmd.format(commit=commit, context_lines=context_lines).split(" "),
            )
            .decode()
            .strip()
            for commit in hashes
        ]
    ]


def get_commits(
    before_ref: str = "origin/main^",
    after_ref: str = "origin/main",
    context_lines: int = 5,
) -> List[Commit]:
    """Get the list of commits between two references"""
    # Get the commit hashes for BEFORE and AFTER
    before_hash = rev_parse(before_ref)
    subprocess.check_call(["git", "fetch"])
    after_hash = rev_parse(after_ref)

    # Get the list of commit hashes between before and after
    hashes: List[str] = (
        subprocess.check_output(
            ["git", "rev-list", "--no-merges", f"{before_hash}..{after_hash}"],
        )
        .decode()
        .splitlines()
    )

    dt_diffs = dt_diffs_from_hashes(hashes, context_lines=context_lines)
    dts = [dt_diff[0] for dt_diff in dt_diffs]
    diffs = [dt_diff[1] for dt_diff in dt_diffs]
    # Return a list of Commit objects
    return [
        Commit(
            commit_hash=commit_hash.strip(),
            date_time_str=date_time_str,
            diff=diff.strip(),
        )
        for commit_hash, date_time_str, diff in zip(
            hashes,
            dts,
            diffs,
        )
    ]


def get_descriptions(
    commits: List[Commit],
    provider: str,
    llm: BaseChatModel,
    prompt: ChatPromptTemplate,
    verbose: bool = True,
    max_concurrency: int = 0,
) -> List[CommitInfo]:
    """Get the descriptions for a list of commits"""
    config_dict: dict[str, Any] = {"verbose": verbose}
    if max_concurrency > 0:
        config_dict["max_concurrency"] = max_concurrency
    outputs: List[CommitDescription]
    if provider == "openai":
        chain = create_structured_output_chain(
            CommitDescription,
            llm,
            prompt,
        )
        results: List[dict] = chain.batch(
            [commit.dict() for commit in commits],
            RunnableConfig(config_dict),
        )
        outputs = [result["function"] for result in results]

    else:
        chain = get_non_openai_chain(llm)

        outputs = chain.batch(
            [{"input": commit.diff} for commit in commits],
            RunnableConfig(config_dict),
        )

    return [
        CommitInfo(**commit.dict(), **commit_description.dict())
        for commit, commit_description in zip(commits, outputs)
    ]


def get_existing_changelog(
    before_ref: str,
    output_file: str = "AI_CHANGELOG.md",
) -> Union[str, None]:
    # Check to see if output_file already exists
    if os.path.isfile(output_file):
        # If so, restore the original version from main
        subprocess.call(["git", "checkout", before_ref, "--", output_file])

        # Get its contents starting from the second line
        with open(output_file, "r") as existing_changelog:
            return "\n".join(
                [line.strip() for line in existing_changelog.readlines()[1:]],
            ).strip()
    return None


def update_changelog(
    before_ref: str,
    new_commits: List[Commit],
    provider: str,
    llm: BaseChatModel,
    prompt: ChatPromptTemplate,
    verbose: bool = True,
    max_concurrency: int = 0,
    output_file: str = "AI_CHANGELOG.md",
) -> None:
    new_commit_infos: List[CommitInfo] = get_descriptions(
        new_commits,
        provider,
        llm,
        prompt,
        verbose,
        max_concurrency,
    )
    new_descriptions: str = CommitInfo.infos_to_str(new_commit_infos).strip()
    existing_content = get_existing_changelog(before_ref, output_file) or ""

    output = f"# AI CHANGELOG\n{new_descriptions.strip()}\n{existing_content.strip()}\n".strip()

    # Write the output to AI_CHANGELOG.md
    with open(output_file, "w") as new_changelog:
        new_changelog.write(output)
