import json
from typing import Callable, List, Any, Dict, TypeVar

import openai
from app import App
from io_utils.discord_io import DiscordIO
from io_utils.telegram_io import TelegramIO
from io_utils.text_file_io import TextFileIO
from io_utils.dummy_io import DummyIO
from summarizer.json_summarizers.gpt_MR_summarizer import GPTMRSummarizer
from summarizer.str_summarizers.chatgpt_summarizer import ChatGPTSummarizer
from summarizer.str_summarizers.flan_t5_base_samsum_summarizer import (
    FlanT5BaseSamsumSummarizer,
)
from summarizer.json_summarizers.tree_summarizer import TreeSummarizer
from summarizer.json_summarizers.json_summarizer import JsonSummarizer, Json
from summarizer.str_summarizers.identity_summarizer import IdentitySummarizer
from io_utils.abstract_io import AbstractIO


class IOFactory:
    @staticmethod
    def new(io: str, **kwargs) -> AbstractIO:
        match io:
            case "Discord":
                return DiscordIO(**kwargs)
            case "Telegram":
                return TelegramIO(**kwargs)
            case "Text File":
                return TextFileIO(**kwargs)
            case "Dummy":
                return DummyIO()
            case _:
                raise ValueError("not supported type of io")


class SummarizerFactory:
    @staticmethod
    def new(summarizer: str, **kwargs) -> JsonSummarizer:
        match summarizer:
            case "ChatGPT recursive":
                return TreeSummarizer(ChatGPTSummarizer(**kwargs))
            case "ChatGPT MapReduce":
                return GPTMRSummarizer(**kwargs)
            case "Identity":
                return TreeSummarizer(IdentitySummarizer())
            case "Flan T5 base samsum":
                return TreeSummarizer(FlanT5BaseSamsumSummarizer(**kwargs))
            case _:
                raise ValueError("not supported type of summarizer")


T = TypeVar("T", Dict[str, Any], List, str)


class AppFactory:
    """define an app using a json file, builds corresponding application"""

    @classmethod
    def dispatch(cls, obj: T) -> T:
        match obj:
            case dict():
                return {k: cls.dispatch(v) for k, v in obj.items()}
            case list():
                return [cls.dispatch(s) for s in obj]
            case str():
                if obj.startswith("file:"):
                    with open(obj[len("file:") :], "r") as f:
                        return f.read()
                return obj

    @classmethod
    def replace_secret(cls, obj: Json) -> Json:
        # replace all string starts with `file:` with file content
        return cls.dispatch(obj)

    @classmethod
    def new(cls, config: Json) -> App:
        input_io = cls.replace_secret(config["input_io"])
        summarizer = cls.replace_secret(config["summarizer"])
        output_io = cls.replace_secret(config["output_io"])
        return App(
            IOFactory.new(input_io["name"], **input_io["args"]),
            SummarizerFactory.new(summarizer["name"], **summarizer["args"]),
            IOFactory.new(output_io["name"], **output_io["args"]),
        )
