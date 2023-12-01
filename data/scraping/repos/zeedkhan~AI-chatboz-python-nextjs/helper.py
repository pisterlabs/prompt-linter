from typing import Any, Callable, TypeVar, Dict

from langchain import BasePromptTemplate, LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseOutputParser, OutputParserException

T = TypeVar("T")


def parse_with_handling(parser: BaseOutputParser[T], completion: str) -> T:
    try:
        return parser.parse(completion)
    except OutputParserException as e:
        raise e


async def openai_error_handler(
    func: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    try:
        return await func(*args, **kwargs)
    except ValueError as e:
        raise e


async def call_model_with_handling(
    model: BaseChatModel,
    prompt: BasePromptTemplate,
    args: Dict[str, str],
    **kwargs: Any,
) -> str:
    chain = LLMChain(llm=model, prompt=prompt)
    return await openai_error_handler(chain.arun, args, **kwargs)
