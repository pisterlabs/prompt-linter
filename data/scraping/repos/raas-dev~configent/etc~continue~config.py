"""
This is the Continue configuration file.

See https://continue.dev/docs/customization to learn more.
"""

import os
import subprocess
from textwrap import dedent

from continuedev.core.config import (
    ContinueConfig,
    CustomCommand,
    SlashCommand,
)
from continuedev.core.main import Step
from continuedev.core.models import Models
from continuedev.core.sdk import ContinueSDK
from continuedev.libs.llm.openai import OpenAI
from continuedev.plugins.context_providers import (
    DiffContextProvider,
    FileTreeContextProvider,
    GitHubIssuesContextProvider,  # noqa: F401
    GoogleContextProvider,
    SearchContextProvider,
    TerminalContextProvider,
    URLContextProvider,
)
from continuedev.plugins.context_providers.file import FileContextProvider
from continuedev.plugins.policies.default import DefaultPolicy
from continuedev.plugins.steps import (
    ClearHistoryStep,
    CommentCodeStep,
    EditHighlightedCodeStep,
    GenerateShellCommandStep,
    OpenConfigStep,
)
from continuedev.plugins.steps.share_session import ShareSessionStep


class CommitMessageStep(Step):
    """
    This is a Step, the building block of Continue.
    It can be used below as a slash command, so that
    run will be called when you type '/commit'.
    """

    async def run(self, sdk: ContinueSDK):
        # Get the root directory of the workspace
        dir = sdk.ide.workspace_directory

        # Run git diff in that directory
        diff = subprocess.check_output(["git", "diff"], cwd=dir).decode("utf-8")

        # Ask the LLM to write a commit message,
        # and set it as the description of this step
        self.description = await sdk.models.default.complete(
            f"{diff}\n\nWrite a short, specific (less than 50 chars) commit message about the above changes:"
        )


API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY") or ""

config = ContinueConfig(
    allow_anonymous_telemetry=False,
    models=Models(
        default=OpenAI(api_key=API_KEY, model="gpt-4-1106-preview"), # type: ignore
        summarize=OpenAI(api_key=API_KEY, model="gpt-4-1106-preview"), # type: ignore
    ),

    system_message="You are a helpful assistant. Please make all responses as concise as possible and never repeat something you have already explained.",

    temperature=0.0,
    custom_commands=[
        CustomCommand(
            name="test",
            description="Write unit tests for the highlighted code",
            prompt="Write a comprehensive set of unit tests for the selected code. It should setup, run tests that check for correctness including important edge cases, and teardown. Ensure that the tests are complete and sophisticated. Give the tests just as chat output, don't edit any file.",
        ),
        CustomCommand(
            name="check",
            description="Check for mistakes in my code",
            prompt=dedent(
                """\
            Please read the highlighted code and check for any mistakes. You should look for the following, and be extremely vigilant:
            - Syntax errors
            - Logic errors
            - Security vulnerabilities
            - Performance issues
            - Anything else that looks wrong

            Once you find an error, please explain it as clearly as possible, but without using extra words. For example, instead of saying "I think there is a syntax error on line 5", you should say "Syntax error on line 5". Give your answer as one bullet point per mistake found."""
            ),
        ),
    ],
    # Slash commands let you run a Step from a slash command
    slash_commands=[
        SlashCommand(
            name="edit",
            description="Edit code in the current file or the highlighted code",
            step=EditHighlightedCodeStep,
        ),
        SlashCommand(
            name="config",
            description="Customize Continue - slash commands, LLMs, system message, etc.",
            step=OpenConfigStep,
        ),
        SlashCommand(
            name="comment",
            description="Write comments for the current file or highlighted code",
            step=CommentCodeStep,
        ),
        SlashCommand(
            name="commit",
            description="Generate a commit message for the current changes",
            step=CommitMessageStep,
        ),
        SlashCommand(
            name="clear",
            description="Clear step history",
            step=ClearHistoryStep,
        ),
        SlashCommand(
            name="share",
            description="Download and share the session transcript",
            step=ShareSessionStep,
        ),
        SlashCommand(
            name="cmd",
            description="Generate a shell command",
            step=GenerateShellCommandStep,
        )
    ],
    # Context providers let you quickly select context by typing '@'
    context_providers=[
        DiffContextProvider(), # type: ignore
        FileContextProvider(), # type: ignore
        FileTreeContextProvider(), # type: ignore
        #GitHubIssuesContextProvider(
        #     repo_name="<your github username or organization>/<your repo name>",
        #     auth_token="<your github auth token>"
        #),
        GoogleContextProvider(serper_api_key=SERPER_API_KEY), # type: ignore
        SearchContextProvider(), # type: ignore
        TerminalContextProvider(), # type: ignore
        URLContextProvider(
            preset_urls=[
                # Add any common urls you reference here so they appear in autocomplete
            ]
        ), # type: ignore
    ],
    # Policies hold the main logic that decides which Step to take next
    # You can use them to design agents, or deeply customize Continue
    policy=DefaultPolicy(),
)
