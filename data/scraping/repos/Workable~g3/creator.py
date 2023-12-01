from typing import Optional

import pyperclip
from rich import print

from g3.domain.message_tone import MessageTone
from g3.services.generate.client import OpenAIChat
from g3.services.generate.commit.prompts.creator import Creator as PromptCreator
from g3.services.generate.preview.cli import Presenter
from g3.services.git import client as git
from g3.services.github.client import Client


class Creator:
    def __init__(self):
        self.prompt_creator = PromptCreator()
        self.openai = OpenAIChat()

    def create(
        self, tone: MessageTone, jira: Optional[str] = None, include: Optional[str] = None, edit: Optional[str] = None
    ) -> None:
        prompt = self.prompt_creator.create(tone, jira, include)

        stream = self.openai.stream(prompt)
        if edit:
            commit = Client().get_commit(commit_hash=edit)
            original_message = commit.commit.message
            reviewed_message, retry = Presenter.present_comparison(original_message, stream, "commit")
            while retry:
                stream = self.openai.stream(prompt)
                reviewed_message, retry = Presenter.present_comparison(original_message, stream, "commit")

            pyperclip.copy(reviewed_message)
            print("✅ Generated message has been copied to clipboard.")
        else:
            reviewed_message, retry = Presenter.present(stream, "commit")
            while retry:
                stream = self.openai.stream(prompt)
                reviewed_message, retry = Presenter.present(stream, "commit")

            git.commit(reviewed_message)
