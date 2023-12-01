from __future__ import annotations

import random
from collections import UserDict, UserList
from pathlib import Path
from typing import Any

from jinja2 import Environment

from clippy.constants import LLM_TASK_BANK_DIR, TASK_BANK_DIR, TASK_BANK_FILE
from clippy.stubs.stubs import StubTemplates

task_base_format = "_base"
word_bank_format = "_bank"
bank_ignore_marker = "#"


def strip_numbering(word: str) -> str:
    """assume format is like:
    `16. Find out the current traffic conditions in {{ wordbank.city }}.`
    """

    if len(words_split := word.split(" ", 1)) > 1:
        numbering, word_rest = words_split
        numbering = numbering.rstrip(".")

        if numbering.isdigit():
            return word_rest

    return word


def process_word_bank(word_bank: str) -> list[str]:
    with open(word_bank) as f:
        words_raw = f.read().splitlines()

    words_raw = [strip_numbering(w) for w in words_raw]

    return words_raw


class Words(UserList):
    def __init__(self, words: list[str] = [], word_var: str = None) -> None:
        super().__init__(words)
        self.word_var = word_var

    @classmethod
    def from_file(cls, file: str | Path):
        if isinstance(file, str):
            file = Path(file)

        word_var = file.name.split(word_bank_format)[0]
        return cls(process_word_bank(file), word_var=word_var)

    def sample(self, seed: int | None = None) -> Any:
        if seed != None:
            return self[seed % len(self)]
        return random.choice(self)


class WordBank(UserDict):
    def __init__(self, _seed: int | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._seed = _seed

    def __getattr__(self, key):
        return self[key].sample(self._seed)


class TaskBankManager:
    def __init__(self, task_bank_dir: str = TASK_BANK_DIR, seed: int = None) -> None:
        self.task_bank_dir = task_bank_dir
        self.tasks = []
        self._seed = seed
        self.wordbank = WordBank(self._seed)

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self) -> TaskBankManager:
        self._task_iter = iter(self.tasks)
        return self

    def __next__(self):
        task = next(self._task_iter)
        return task.render(wordbank=self.wordbank)

    def setup(self) -> TaskBankManager:
        self.process_all_word_banks()

        self.task_templates_file = f"{self.task_bank_dir}/task{task_base_format}"
        with open(self.task_templates_file) as f:
            self.tasks_raw = f.read().splitlines()

        self.env = Environment()
        for task in self.tasks_raw:
            template = self.env.from_string(task)
            self.tasks.append(template)
        return self

    def process_all_word_banks(self) -> None:
        for file in Path(self.task_bank_dir).iterdir():
            if file.name.endswith(word_bank_format):
                words = Words.from_file(file)
                self.wordbank[words.word_var] = words

    def sample(self, idx: int | None = None) -> str:
        if idx is None:
            idx = random.randint(0, len(self.tasks) - 1)

        return self.tasks[idx].render(wordbank=self.wordbank)


class LLMTaskGenerator:
    """
    This class represents a task generator for the LLM model.
    It is responsible for generating tasks based on a task bank file.
    """

    def __init__(
        self,
        task_bank_dir: str = LLM_TASK_BANK_DIR,
        task_bank_file: str = TASK_BANK_FILE,
        client: "Controller" = None,
        seed: int = None,
    ) -> None:
        self.task_templates_file = Path(task_bank_dir)
        if task_bank_file:
            self.task_templates_file /= task_bank_file

        self._seed = seed
        self._sync_client = client

        self.task_templates = []

        with open(self.task_templates_file) as f:
            self.task_templates_raw = f.read().splitlines()

        self.task_templates_raw = [t for t in self.task_templates_raw if not t.startswith(bank_ignore_marker)]

    def process_response(self, resp: "Generations") -> str:
        """
        Process the response from the LLM model and extract the generated text.
        """
        text = resp.generations[0].text
        text = text.lstrip(" ")
        return text

    def setup(self, sync_client: "Controller" = None):
        """
        Set up the LLM task generator with a synchronous client.
        """
        if sync_client is None:
            from clippy.controllers.apis.cohere_controller import CohereController

            sync_client = CohereController.get_client(client_type="sync")

        self._sync_client = sync_client

        return self

    def get_prompt(self) -> tuple[str, str]:
        """
        Get a random prompt from the task templates and return it along with the raw task.
        """
        raw_task = random.choice(self.task_templates_raw)
        prompt = StubTemplates.task_gen.render(task_template=raw_task)
        return prompt, raw_task

    def sample_sync(
        self,
        client: "Controller" = None,
        max_tokens: int = 200,
        stop_sequences: list[str] = ["\n", "--", "---"],
        generate_kwargs: dict[str, Any] = {},
        return_raw_task: bool = False,
    ) -> str:
        """
        Generate a task synchronously using the LLM model.
        """
        client = client or self._sync_client
        prompt, raw_task = self.get_prompt()
        response = client.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            **generate_kwargs,
        )
        text = self.process_response(response)

        # allow the raw task to be returned
        return text if not return_raw_task else (text, raw_task)

    async def sample(
        self,
        client: "Controller" = None,
        max_tokens: int = 200,
        stop_sequences: list[str] = ["\n", "--", "---"],
        generate_kwargs: dict[str, Any] = {},
        return_raw_task: bool = False,
    ) -> str:
        """
        Generate a task asynchronously using the LLM model.
        """
        client = client or self.client
        prompt, raw_task = self.get_prompt()
        response = await client.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            **generate_kwargs,
        )

        text = self.process_response(response)
        return text if not return_raw_task else (text, raw_task)
