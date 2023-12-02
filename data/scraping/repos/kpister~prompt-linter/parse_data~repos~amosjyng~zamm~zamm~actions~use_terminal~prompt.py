from typing import List

from langchain.prompts.prompt import PromptTemplate
from langchain_contrib.prompts import PrefixedTemplate

TerminalPromptTemplate = PrefixedTemplate(
    PromptTemplate.from_template(
        """
You proceed to use the terminal:

```bash
$ """.lstrip()
    )
)

TERMINAL_USAGE_PREFIX = """
You proceed to use the terminal:

```bash"""

TERMINAL_USAGE_SUFFIX = "```"


class TerminalUsageLogger(PromptTemplate):
    input_variables: List[str] = ["command", "output"]
    template: str = """
$ {command}
{output}""".lstrip()
