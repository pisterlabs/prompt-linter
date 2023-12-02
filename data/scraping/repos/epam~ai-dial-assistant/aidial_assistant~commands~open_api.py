from typing import Any, List

from langchain.tools.openapi.utils.api_models import APIOperation
from typing_extensions import override

from aidial_assistant.commands.base import (
    Command,
    ExecutionCallback,
    ResultObject,
)
from aidial_assistant.open_api.requester import OpenAPIEndpointRequester


class OpenAPIChatCommand(Command):
    @staticmethod
    def token() -> str:
        return "open-api-chat-command"

    def __init__(self, op: APIOperation, plugin_auth: str | None):
        self.op = op
        self.plugin_auth = plugin_auth

    @override
    async def execute(
        self, args: List[Any], execution_callback: ExecutionCallback
    ) -> ResultObject:
        self.assert_arg_count(args, 1)

        return await OpenAPIEndpointRequester(
            self.op, self.plugin_auth
        ).execute(args[0])
