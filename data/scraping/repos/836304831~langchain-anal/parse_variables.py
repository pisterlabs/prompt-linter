# -*- coding: utf-8 -*-
# ----------------------------
# @Time    : 2023/7/15 11:10
# @Author  : acedar
# @FileName: parse_variables.py
# ----------------------------

from jinja2 import Environment, meta
from string import Formatter


from string import Formatter
from typing import Any, List, Mapping, Sequence, Union


class StrictFormatter(Formatter):
    """A subclass of formatter that checks for extra keys."""

    def check_unused_args(
        self,
        used_args: Sequence[Union[int, str]],
        args: Sequence,
        kwargs: Mapping[str, Any],
    ) -> None:
        """Check to see if extra parameters are passed."""
        extra = set(kwargs).difference(used_args)
        if extra:
            raise KeyError(extra)

    def vformat(
        self, format_string: str, args: Sequence, kwargs: Mapping[str, Any]
    ) -> str:
        """Check that no arguments are provided."""
        if len(args) > 0:
            raise ValueError(
                "No arguments should be provided, "
                "everything should be passed as keyword arguments."
            )
        return super().vformat(format_string, args, kwargs)

    def validate_input_variables(
        self, format_string: str, input_variables: List[str]
    ) -> None:
        dummy_inputs = {input_variable: "foo" for input_variable in input_variables}
        super().format(format_string, **dummy_inputs)

formatter = StrictFormatter()

def variables_from_template(template, **kwargs):
    if "template_format" in kwargs and kwargs["template_format"] == "jinja2":
        # Get the variables for the template
        env = Environment()
        ast = env.parse(template)
        variables = meta.find_undeclared_variables(ast)
    else:
        variables = {
            v for _, v, _, _ in Formatter().parse(template) if v is not None
        }
    return variables


template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""

variables = variables_from_template(template_string)

print(variables)

# from langchain.formatting import formatter


kwargs = {
    "style": """American English \
                in a calm and respectful tone
                """,
    "text": """Arrr, I be fuming that me blender lid \
                flew off and splattered me kitchen walls \
                with smoothie! And to make matters worse, \
                the warranty don't cover the cost of \
                cleaning up me kitchen. I need yer help \
                right now, matey!"""
}

result = formatter.format(template_string, **kwargs)
print("result:\n", result)
