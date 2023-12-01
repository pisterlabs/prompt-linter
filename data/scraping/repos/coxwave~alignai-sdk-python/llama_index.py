import importlib
from datetime import datetime
from importlib import metadata as importlib_metadata
from typing import TypedDict

from packaging import version

from alignai import AlignAI
from alignai.constants import DEFAULT_ASSISTANT_ID, ROLE_ASSISTANT, ROLE_USER

try:
    llama_index = importlib.import_module("llama_index")
    llama_index_version = importlib_metadata.version("llama_index")
    if version.parse(llama_index_version) < version.parse("0.8.4"):
        raise ImportError("You must install llama_index>=0.8.4 to use Align AI integrated ChatMemory.")
    BaseMemory = importlib.import_module("llama_index.memory.types").BaseMemory
    ChatMemoryBuffer = importlib.import_module("llama_index.memory.chat_memory_buffer").ChatMemoryBuffer
    llm_base = importlib.import_module("llama_index.llms.base")
    MessageRole = llm_base.MessageRole
    ChatMessage = llm_base.ChatMessage
    LLM = llm_base.LLM
except ModuleNotFoundError:
    raise ImportError("You must install 'llama_index' to use Align AI integrated ChatMemory.")


class UserInfo(TypedDict, total=False):
    email: str | None
    ip: str | None
    country_code: str | None
    create_time: datetime | None
    display_name: str | None


class ChatMemory(BaseMemory):
    chat_memory: BaseMemory
    align_client: AlignAI
    session_id: str
    user_id: str
    assistant_id: str = DEFAULT_ASSISTANT_ID
    user_info: UserInfo | None = None
    dynamic_attrs: dict

    class Config:  # for custom type AlignAI against pydantic validation.
        arbitrary_types_allowed = True

    def __init__(
        self,
        chat_memory: BaseMemory,
        align_client: AlignAI,
        session_id: str,
        user_id: str,
        assistant_id: str = DEFAULT_ASSISTANT_ID,
        user_info: UserInfo | None = None,
    ):
        """Initialize ChatMemory. When initialized, open_session event is emitted. If user_info is provided, identify_user event is also emitted.

        Args:
            chat_memory (BaseMemory): Chat memory from LlamaIndex. Reference: https://docs.llamaindex.ai/en/stable/api_reference/memory.html
            align_client (AlignAI): Align AI SDK client.
            session_id (str): Session ID.
            user_id (str): User Id.
            assistant_id (str, optional): Assistant ID. Defaults to "DEFAULT".
            user_info (UserInfo | None, optional): Providing at least one user information will trigger identify_user event upon initialization. Defaults to None.
        """  # noqa: E501
        super().__init__(
            chat_memory=chat_memory,
            align_client=align_client,
            session_id=session_id,
            user_id=user_id,
            assistant_id=assistant_id,
            user_info=user_info,
            dynamic_attrs={},
        )

        if isinstance(chat_memory, ChatMemoryBuffer):
            self.dynamic_attrs["token_limit"] = chat_memory.token_limit
            self.dynamic_attrs["tokenizer_fn"] = chat_memory.tokenizer_fn
            self.dynamic_attrs["chat_history"] = chat_memory.chat_history

        self.align_client.open_session(session_id=self.session_id, user_id=self.user_id, assistant_id=self.assistant_id)
        if self.user_info:
            self.identify_user(
                email=self.user_info.get("email", None),
                ip=self.user_info.get("ip", None),
                country_code=self.user_info.get("country_code", None),
                create_time=self.user_info.get("create_time", None),
                display_name=self.user_info.get("display_name", None),
            )

    def __getattr__(self, name):
        if name in self.dynamic_attrs:
            return self.dynamic_attrs[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    @classmethod
    def from_defaults(cls, chat_history: list[ChatMessage] | None = None, llm: LLM | None = None):
        """ChatMemory does not support 'from_defaults' constructor. Please use the constructor directly."""
        raise NotImplementedError(
            "from_defaults is not intended to be used with ChatMemory. Please use the constructor directly."
        )

    def get(self, **kwargs: any) -> list[ChatMessage]:
        """Get chat history.

        Returns:
            list[ChatMessage]: Chat history.
        """
        return self.chat_memory.get(**kwargs)

    def get_all(self) -> list[ChatMessage]:
        """Get all chat history.

        Returns:
            list[ChatMessage]: Chat history.
        """
        return self.chat_memory.get_all()

    def put(self, message: ChatMessage) -> None:
        """Put chat history. If the message is user or assistant message, create_message event will be emitted to Align AI.

        Args:
            message (ChatMessage): Chat message.
        """  # noqa: E501
        if message.role in [MessageRole.USER, MessageRole.ASSISTANT] and message.content is not None:
            self.align_client.create_message(
                session_id=self.session_id,
                message_index=self._next_message_idx,
                role=ROLE_USER if message.role == MessageRole.USER else ROLE_ASSISTANT,
                content=message.content,
            )
        self.chat_memory.put(message)

    def set(self, messages: list[ChatMessage]) -> None:
        """Set chat history.

        Args:
            messages (list[ChatMessage]): Chat history to be set.
        """
        self.chat_memory.set(messages)

    def reset(self) -> None:
        """Reset chat history."""
        self.chat_memory.reset()

    def identify_user(
        self,
        email: str | None = None,
        ip: str | None = None,
        country_code: str | None = None,
        create_time: datetime | None = None,
        display_name: str | None = None,
    ) -> None:
        """Send identify_user event to Align AI. The user_id provided upon initialization will be used.

        Args:
            email (str | None, optional): User email address. Defaults to None.
            ip (str | None, optional): User IPv4 address. Provide either ip or country code for user location. If both are given, country code overrides ip. Defaults to None.
            country_code (str | None, optional): User country code in ISO Alpha-2. Provide either ip or country code for user location. If both are given, country code overrides ip. Defaults to None.
            create_time (datetime | None, optional): User creation time. Defaults to None.
            display_name (str | None, optional): User display name. Defaults to None.
        """  # noqa: E501
        self.align_client.identify_user(
            user_id=self.user_id,
            email=email,
            ip=ip,
            country_code=country_code,
            create_time=create_time,
            display_name=display_name,
        )

    def close(self) -> None:
        """Close the session. Do not send additional message after calling this method."""
        self.align_client.close_session(session_id=self.session_id)

    @property
    def _messages(self) -> list[ChatMessage]:
        return self.get_all()

    @property
    def _next_message_idx(self) -> int:
        return (
            len([message for message in self._messages if message.role in [MessageRole.USER, MessageRole.ASSISTANT]])
            + 1
        )
