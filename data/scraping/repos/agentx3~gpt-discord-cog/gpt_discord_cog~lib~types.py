from typing import (
    Callable,
    TypedDict,
)
from discord import ApplicationContext

from openai import AsyncOpenAI
import sqlite3


class CommandOptions(TypedDict):
    # This should be compatible with the rules for a discord command name
    name: str
    description: str
    check: Callable[[ApplicationContext], bool]


class OptionalCommandOptions(TypedDict, total=False):
    name: str
    description: str
    check: Callable[[ApplicationContext], bool]


class CommandConfig(TypedDict):
    modify: CommandOptions


class OptionalCommandConfig(TypedDict, total=False):
    modify: OptionalCommandOptions


class ImageConfig(TypedDict):
    enable: bool


class OptionalConfig(TypedDict, total=False):
    commands: OptionalCommandConfig
    image: ImageConfig


class RequiredGPTConfig(TypedDict):
    client: AsyncOpenAI
    assistant_id: str
    database_connection: sqlite3.Connection
    database_name: str
    conversation_lifetime: int


class GPTConfig(RequiredGPTConfig):
    commands: CommandConfig
    image: ImageConfig


class UserGPTConfig(RequiredGPTConfig, OptionalConfig, total=False):
    pass
