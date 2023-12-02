from typing import Literal, Optional

import requests
from langchain.pydantic_v1 import BaseModel, Field

from jarvis.utils import log

from .. import Gadget

_name = "ha_api_client"

_documentation = """\
"""

with open("ha_api_client_documentation", "r+") as docs_file:
    _documentation = "".join(docs_file.readlines())

_usage_instructions = f"""\
Utilize the {_name} gadget when you would like to execute HTTP requests \
against a running instance of the home automation system which is connected to \
several home automation devices. By interacting with the system via HTTP \
requests you can manipulate the devices themselves. leverage the documentation
to understand what verbs and endpoints are available and how to call them.

When leveraging this gadget you MUST ALWAYS utilize the following algorithm:
1. Discover devices connected to the home automation system.
2. access if the intent in question refers to one of the discovered devices.
3. execute a state change to accomplish the users intent.
4. Scan the system again to see if we can verify the intents accomplishment.
"""


class HAHttpRequestAction(BaseModel):
    name: Literal["ha_api_client"] = Field(
        description="Use this kind when you wish to make an HTTP request "
        + f"to the {_name} gadget. No need to worry about the scheme, or domain portion "
        + "of an HTTP request."
    )
    verb: str = Field(description="Verb to be used in the HTTP request.")
    path: str = Field(description="Path portion of the HTTP request.")
    query: str = Field(default="", description="query string portion of the HTTP request.")
    body: Optional[str] = Field(description="Body of the HTTP request.")


class HAAPIClient(Gadget):
    scheme: str
    domain: str
    token: str

    def __init__(self, scheme: str, domain: str, token: str):
        self.scheme = scheme
        self.domain = domain
        self.token = token

    @log
    def req(self, action: HAHttpRequestAction) -> requests.Response:
        url = f"{self.scheme}://{self.domain}{action.path}?{action.query}"
        return requests.request(
            method=action.verb,
            url=url,
            data=action.body,
            headers={
                "authorization": "Bearer {}".format(self.token),
                "content-type": "application/json",
            },
        )

    def get_name(self) -> str:
        return _name

    def get_documentation(self) -> str:
        return _documentation

    def get_usage_instructions(self) -> str:
        return _usage_instructions
