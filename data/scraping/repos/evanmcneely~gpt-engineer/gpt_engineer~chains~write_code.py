from halo import Halo
import re
from typing import List, Tuple
from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import StreamingStdOutCallbackHandler

from ..llm import get_llm
from config import Models

template = """
Please now remember the steps:

Think step by step and reason yourself to the right decisions to make sure we get it right.
First lay out the names of the core classes, functions, methods that will be necessary, As well as a quick comment on their purpose.

Then you will output the content of each file including ALL code.
Each file must strictly follow a markdown code block format, where the following tokens must be replaced such that
FILENAME is the lowercase file name including the file extension,
LANG is the markup code block language for the code's language, and CODE is the code:

FILENAME
```LANG
CODE
```

Please note that the code should be fully functional. No placeholders.


Chat history:
{chat_history}

Begin
"""


def _codeblock_search(chat: str) -> re.Match:
    regex = r"(\S+)\n\s*```[^\n]*\n(.+?)```"
    return re.finditer(regex, chat, re.DOTALL)


def _parse_chat(chat) -> List[Tuple[str, str]]:
    matches = _codeblock_search(chat)

    files = []
    for match in matches:
        # Strip the filename of any non-allowed characters and convert / to \
        path = re.sub(r'[<>"|?*]', "", match.group(1))

        # Remove leading and trailing brackets
        path = re.sub(r"^\[(.*)\]$", r"\1", path)

        # Remove leading and trailing backticks
        path = re.sub(r"^`(.*)`$", r"\1", path)

        # Remove trailing ]
        path = re.sub(r"\]$", "", path)

        # Get the code
        code = match.group(2)

        # Add the file to the list
        files.append((path, code))

    # Get all the text before the first ``` block
    readme = chat.split("```")[0]
    files.append(("README.md", readme))

    # Return the files
    return files


@Halo(text="Generating code", spinner="dots")
def write_code(memory: str):
    chain = LLMChain(
        llm=get_llm(Models.CODE_MODEL),
        prompt=PromptTemplate.from_template(template),
    )

    result = chain.predict(chat_history=memory)

    return _parse_chat(result)
