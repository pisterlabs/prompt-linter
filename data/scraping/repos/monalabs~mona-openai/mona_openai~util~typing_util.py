from typing import TYPE_CHECKING, Type, Union

if TYPE_CHECKING:
    import openai

CompletionType = Type["openai.api_resources.completion.Completion"]
ChatCompletionType = Type[
    "openai.api_resources.chat_completion.ChatCompletion"
]
SupportedOpenAIClassesType = Union[CompletionType, ChatCompletionType]
