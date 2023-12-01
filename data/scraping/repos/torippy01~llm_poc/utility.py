import langchain
import re

from typing import TypeVar, Optional

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from mdutils.mdutils import MdUtils
from openai import ChatCompletion

## you can use typing.Self after python 3.11
Self = TypeVar("Self")


def set_up() -> None:
    load_dotenv()
    langchain.verbose = True
    return


def get_gpt_response(query: str) -> str:
    response = ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": query}]
    )
    return response["choices"][0]["message"]["content"].strip()


"""
def time_measurement(func: Callable, val: Any) -> Any:
    start = time.time()
    response = func(**val)
    elapsed_time = time.time() - start
    return response, elapsed_time
"""


def create_llm(llm_name: str) -> ChatOpenAI:
    return ChatOpenAI(temperature=0, model_name=llm_name)


def create_CBmemory() -> ConversationBufferMemory:
    return ConversationBufferMemory(
        return_messages=True, memory_key="chat_history", output_key="output"
    )


def sep_md(mdFile: MdUtils) -> None:
    mdFile.new_line()
    mdFile.new_line("---")
    mdFile.new_line()


def host_validation(host: Optional[str]):
    # hostが文字列であればTrue
    # TODO: 文字列の内容を加味すべき
    if not host:
        return False
    elif isinstance(host, str):
        return True


def port_validation(port: Optional[str]):
    # portが半角数字文字列であればTrue
    # それ以外はFalse
    if not port:
        return False
    return True if re.fullmatch("[0-9]+", port) else False
