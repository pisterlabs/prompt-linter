import re
from dataclasses import dataclass, field
from functools import total_ordering
from typing import List, TypedDict, Tuple

_code_pattern = re.compile(r"(\d+)(\w*)")


def _get_sortable_name(value) -> Tuple[int, str]:
    match = _code_pattern.match(value)
    if match is None:
        return 9999999, value
    else:
        return int(match.group(1)), match.group(2)


@dataclass
@total_ordering
class ErrorDefinition:
    """
    Error definition information that is passed onto the frontend tool. The code and description are used for display, 
    and the affected_fields is a list of fields which will be highlighted by the frontend tool if present.

    :param code: String describing the error code e.g. '103'
    :param description: String describing the error (from guidance) e.g. 'The ethnicity code is either not valid or has not been entered'
    :param affected_fields: A list of fields to highlight in the tool e.g. ['ETHNIC']
    """
    code: str
    description: str
    affected_fields: List[str]
    sortable_code: Tuple[int, str] = field(init=False)

    def __post_init__(self):
        self.sortable_code = _get_sortable_name(self.code)

    @staticmethod
    def _is_valid_operand(other):
        return (
            hasattr(other, "code")
            and
            hasattr(other, "sortable_code")
        )

    def __eq__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return self.code == other.code

    def __lt__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return self.sortable_code < other.sortable_code


class UploadedFile(TypedDict):
    name: str
    fileText: bytes
    description: str


class UploadException(Exception):
    pass

