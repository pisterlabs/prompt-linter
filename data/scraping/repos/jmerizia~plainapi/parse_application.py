from typing import TypedDict
import os
from dotenv import load_dotenv
import openai  # type: ignore

from plainapi.parse_endpoint import parse_endpoint, Endpoint


class Application(TypedDict):
    title: str
    endpoints: list[Endpoint]


def parse_application(endpoints_code: str, functions_code: str, schema_text: str) -> Application:
    blocks = [s.strip() for s in endpoints_code.split('\n\n') if s.strip() != '']
    if len(blocks) < 1:
        raise ValueError('Expected at least one block in the endpoints file (for the title).')
    title_block = blocks[0]
    title = title_block.split('\n')[0].strip()
    endpoints: list[Endpoint] = []
    for block in blocks[1:]:
        offset = 1  # just being lazy
        endpoint = parse_endpoint(endpoint_string=block, schema_text=schema_text, global_line_offset=offset)
        endpoints.append(endpoint)
    return {
        'title': title,
        'endpoints': endpoints
    }

