import json
import logging
from typing import Dict, List, NamedTuple, Optional

import aiohttp.client_exceptions
from aiohttp import hdrs
from langchain.tools.openapi.utils.api_models import APIOperation

from aidial_assistant.commands.base import JsonResult, ResultObject, TextResult
from aidial_assistant.utils.requests import arequest

logger = logging.getLogger(__name__)


class _ParamMapping(NamedTuple):
    """Mapping from parameter name to parameter value."""

    query_params: List[str]
    body_params: List[str]
    path_params: List[str]


class OpenAPIEndpointRequester:
    """Chain interacts with an OpenAPI endpoint using natural language.
    Based on OpenAPIEndpointChain from LangChain.
    """

    def __init__(self, operation: APIOperation, plugin_auth: str | None):
        self.operation = operation
        self.param_mapping = _ParamMapping(
            query_params=operation.query_params,  # type: ignore
            body_params=operation.body_params,  # type: ignore
            path_params=operation.path_params,  # type: ignore
        )
        self.plugin_auth = plugin_auth

    def _construct_path(self, args: Dict[str, str]) -> str:
        """Construct the path from the deserialized input."""
        path = self.operation.base_url.rstrip("/") + self.operation.path  # type: ignore
        for param in self.param_mapping.path_params:
            path = path.replace(f"{{{param}}}", str(args.pop(param, "")))
        return path

    def _extract_query_params(self, args: Dict[str, str]) -> Dict[str, str]:
        """Extract the query params from the deserialized input."""
        query_params = {}
        for param in self.param_mapping.query_params:
            if param in args:
                query_params[param] = args.pop(param)
        return query_params

    def _extract_body_params(
        self, args: Dict[str, str]
    ) -> Optional[Dict[str, str]]:
        """Extract the request body params from the deserialized input."""
        body_params = None
        if self.param_mapping.body_params:
            body_params = {}
            for param in self.param_mapping.body_params:
                if param in args:
                    body_params[param] = args.pop(param)
        return body_params

    def deserialize_json_input(self, args: dict) -> dict:
        """Use the serialized typescript dictionary.

        Resolve the path, query params dict, and optional requestBody dict.
        """
        path = self._construct_path(args)
        body_params = self._extract_body_params(args)
        query_params = self._extract_query_params(args)
        return {
            "url": path,
            "json": body_params,
            "params": query_params,
        }

    async def execute(
        self,
        args: dict,
    ) -> ResultObject:
        request_args = self.deserialize_json_input(args)
        headers = (
            None
            if self.plugin_auth is None
            else {hdrs.AUTHORIZATION: self.plugin_auth}
        )
        logger.debug(f"Request args: {request_args}")
        async with arequest(
            self.operation.method.value, headers=headers, **request_args  # type: ignore
        ) as response:
            if response.status != 200:
                try:
                    return JsonResult(json.dumps(await response.json()))
                except aiohttp.ContentTypeError:
                    method_str = str(self.operation.method.value)  # type: ignore
                    error_object = {
                        "reason": response.reason,
                        "status_code": response.status,
                        "method:": method_str.upper(),
                        "url": request_args["url"],
                        "params": request_args["params"],
                    }
                    return JsonResult(json.dumps(error_object))

            if "text" in response.headers[hdrs.CONTENT_TYPE]:
                return TextResult(await response.text())

            return JsonResult(json.dumps(await response.json()))
