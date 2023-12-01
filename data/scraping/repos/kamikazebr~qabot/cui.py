import asyncio
import itertools
import sys
from enum import Enum
from typing import AsyncContextManager, Optional

from langchain.schema import LLMResult, ChatResult

from server.callback import InfoChunk, TextChunk
from ui.base import BaseHumanUserInterface
from utils.util import get_total_tokens


class Color(Enum):
    """Color codes for the commandline"""

    BLACK = "\033[30m"  # (Text) Black
    RED = "\033[31m"  # (Text) Red
    GREEN = "\033[32m"  # (Text) Green
    YELLOW = "\033[33m"  # (Text) Yellow
    BLUE = "\033[34m"  # (Text) Blue
    MAGENTA = "\033[35m"  # (Text) Magenta
    CYAN = "\033[36m"  # (Text) Cyan
    WHITE = "\033[37m"  # (Text) White
    COLOR_DEFAULT = "\033[39m"  # Reset text color to default


class CommandlineUserInterface(BaseHumanUserInterface):
    """Commandline user interface."""

    def get_user_input(self) -> str:
        """Get user input and return the result as a string"""
        user_input = input("Input:")
        return str(user_input)

    def get_binary_user_input(self, prompt: str) -> bool:
        """Get a binary input from the user and return the result as a bool"""
        yes_patterns = ["y", "yes", "yeah", "yup", "yep"]
        no_patterns = ["n", "no", "nah", "nope"]
        while True:
            response = input(prompt + " (y/n) ").strip().lower()
            if response in yes_patterns:
                return True
            elif response in no_patterns:
                return False
            else:
                # self.notify(
                #     "Invalid input", "Please enter y or n.", title_color=Color.RED
                # )
                print("Invalid input", "Please enter y or n.")
                continue

    async def notify(
            self,
            message: str,
            title: Optional[str] = None,
            title_color: str | Color = Color.YELLOW,
            stream: bool = False,
    ) -> None:
        """Print a notification to the user"""
        if stream:
            await self.stream(title=title, message=message)
        if isinstance(title_color, str):
            try:
                title_color = Color[title_color.upper()]
            except KeyError:
                raise ValueError(f"{title_color} is not a valid Color")
        self._print_message(title=title, message=message, title_color=title_color)

    async def stream(self, message: str, title: Optional[str] = None):
        """Print a notification to the user"""
        await self._call_callback_text(f"{f'{title}: ' if title else ''}{message}")

    async def _call_callback_text(self, message: str):
        if self.callback is not None:
            await self.callback.on_llm_new_token(TextChunk(token=f"{message}\n"))
            await asyncio.sleep(0.05)

    async def call_callback_info(self, count_tokens: int, model_name: str | None = None):
        if self.callback is not None:
            await self.callback.on_llm_new_token(InfoChunk(count_tokens=count_tokens, model_name=model_name))
            await asyncio.sleep(0.05)

    async def call_callback_info_llm_result(self, llm_result: LLMResult | ChatResult):
        await self.call_callback_info(count_tokens=get_total_tokens(llm_result),
                                      model_name=llm_result.llm_output["model_name"])

    async def call_callback_end(self):
        if self.callback is not None:
            await self.callback.on_llm_end(response=None)
            await asyncio.sleep(0.05)

    async def call_callback_error(self, error: BaseException | KeyboardInterrupt):
        if self.callback is not None:
            await self.callback.on_llm_error(error=error)
            await asyncio.sleep(0.05)

    async def loading(
            self,
            message: str = "Thinking...",
            delay: float = 0.1,
    ) -> AsyncContextManager:
        """Return a context manager that will display a loading spinner"""

        await self._call_callback_text(message)

        return self.Spinner(message=message, delay=delay)

    def _print_message(self, message: str, title_color: Color, title: Optional[str] = None) -> None:
        print(
            f"{f'{title_color.value}{title}{Color.COLOR_DEFAULT.value}:' if title else ''} {message}"
        )

    class Spinner(AsyncContextManager):
        """A simple spinner class"""

        def __init__(self, message="Loading...", delay=0.1):
            """Initialize the spinner class"""
            self.spinner = itertools.cycle(["-", "/", "|", "\\"])
            self.delay = delay
            self.message = message
            self.running = False
            self.task = None

        async def spin(self):
            """Spin the spinner"""
            while self.running:
                sys.stdout.write(next(self.spinner) + " " + self.message + "\r")
                sys.stdout.flush()
                await asyncio.sleep(self.delay)
                sys.stdout.write("\b" * (len(self.message) + 2))

        async def __aenter__(self):
            """Start the spinner"""
            print("aenter")
            self.running = True
            self.task = asyncio.create_task(self.spin())
            await self.task
            return self

        async def __aexit__(self, exc_type, exc_value, exc_traceback):
            """Stop the spinner"""
            print("aexit")
            self.running = False
            self.task.cancel()
            sys.stdout.write("\r" + " " * (len(self.message) + 2) + "\r")
            sys.stdout.flush()
