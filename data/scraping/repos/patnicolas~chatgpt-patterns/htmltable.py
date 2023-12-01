__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2022, 23. All rights reserved."

from typing import Any, AnyStr, Dict, List, Optional, Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class HTMLFormatInput(BaseModel):
    """ Wraps the input for loading the contractor list with condition"""
    input_list: List[Dict[AnyStr, Any]] = Field(..., description="Input list of dictionaries, key, value pairs")


class HTMLTable(BaseTool):
    name = "format_html"
    description = "Useful to display a list of key-value pairs in an HTML table"

    def _run(self, input_list: List[Dict[AnyStr, Any]]) -> AnyStr:
        html_table = format_html(input_list)
        return html_table

    def _arun(self, input_list: List[Dict[AnyStr, Any]]) -> AnyStr:
        raise NotImplementedError("format_html does not support async")

    args_schema: Optional[Type[BaseModel]] = HTMLFormatInput


def format_html(input_list: List[Dict[AnyStr, Any]]) -> AnyStr:
    from src.display import DisplayFormat
    table_html = DisplayFormat.html_table(input_list) if len(input_list) > 0 else "<h4>Empty list</h4>"
    return DisplayFormat.html_insert('input/request.html', table_html)


if __name__ == '__main__':
    import re
    from inspect import cleandoc
    print(cleandoc("""
              Title
                 - 1
                 - 2"""))

    text = "The quick fox brown fox jump over the lazy dog"
    pattern = r"\b\w{4}\b"
    matches = re.findall(pattern, text)
    print(str(matches))

    """
    _input_list = [
        {'contractor': 'Patrick', 'specialty': 'plumbing','location': 'San Mateo'},
        {'contractor': 'John', 'specialty': 'electrical', 'location': 'San Jose'}
    ]
    table_html_str = format_html(_input_list)
    print(table_html_str)
    """
