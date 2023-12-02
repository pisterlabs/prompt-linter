import sys
import os
import traceback
import bot_config
import requests
import json
from datetime import datetime, timedelta
from requests.utils import requote_uri
import urllib

# import pytz
from typing import Any, Dict, Optional, Type

sys.path.append("/root/projects")
import common.bot_logging
from common.bot_comms import (
    publish_event_card,
    publish_list,
    send_to_another_bot,
    send_to_user,
    send_to_me,
    publish_error,
)
from common.bot_utils import tool_description, tool_error

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools import BaseTool
from langchain.utilities import GoogleSearchAPIWrapper

# tool_logger = common.bot_logging.logging.getLogger('ToolLogger')
# tool_logger.addHandler(common.bot_logging.file_handler)


class ERPGETLIST(BaseTool):
    parameters = []
    optional_parameters = []
    name = "ERP_GET_LIST"
    summary = """useful when you need to get a list of documents"""
    parameters.append(
        {
            "name": "doctype",
            "description": "type of documents you wish to retrieve. enclose in double quotes",
        }
    )
    optional_parameters.append(
        {
            "name": "filters",
            "description": 'optional filters to apply to the query. enclose in double quotes, example ["first_name","like","%Jane%"]',
        }
    )
    description = tool_description(name, summary, parameters, optional_parameters)
    return_direct = False

    def _run(
        self,
        doctype: str,
        filters: str = None,
        publish: str = "True",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            endpoint = f"/api/resource/{doctype}"
            endpoint = requote_uri(endpoint)
            if filters:
                endpoint = f"{endpoint}?filters=[{filters}]"

            response = _send_request("GET", endpoint)
            common.bot_logging.bot_logger.info(str(response))

            # Check if response is not None and has a 'data' key
            if response and "data" in response:
                # Convert the API response to string
                string_response = json.dumps(response["data"])
                common.bot_logging.bot_logger.info(string_response)

                # if string_response == "[]" and filters:
                #     # endpoint = f"/resource/Teams%20User"
                #     endpoint = f"/api/resource/DocType/{doctype}"
                #     endpoint = requote_uri(endpoint)
                #     response = _send_request("GET", endpoint)
                #     common.bot_logging.bot_logger.info(str(response))

                #     # Check if response is not None and has a 'data' key
                #     if response and "data" in response:
                #         # Convert the API response to string
                #         string_response = json.dumps(response["data"])
                #         string_response = f"No records returned, try filtering on a different field: {string_response}"

            else:
                string_response = f"{response}"

            return string_response
        except Exception as e:
            # traceback.print_exc()
            tb = traceback.format_exc()
            publish_error(e, tb)
            return tool_error(e, tb, self.description)

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("GET_CALENDAR_EVENTS does not support async")


class ERPGETDOC(BaseTool):
    parameters = []
    optional_parameters = []
    name = "ERP_GET_DOC"
    summary = """useful when you need to get a specific document. Do not supply a filter. Only a docname"""
    parameters.append(
        {
            "name": "doctype",
            "description": "type of document you wish to retrieve. enclose in double quotes",
        }
    )
    parameters.append(
        {
            "name": "docname",
            "description": "document name you wish to retrieve. enclose in double quotes",
        }
    )

    description = tool_description(name, summary, parameters, optional_parameters)
    return_direct = False

    def _run(
        self,
        doctype: str,
        docname: str = None,
        filters: str = None,
        publish: str = "True",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            # Bot insists on using filters

            if filters:
                return tool_error(
                    "Do not use filters, only docname", "", self.description
                )

            if not docname:
                return tool_error(
                    "Missing docname, please supply docname and not filters",
                    "",
                    self.description,
                )

            endpoint = f"/api/resource/{doctype}/{docname}"
            endpoint = requote_uri(endpoint)
            response = _send_request("GET", endpoint)
            common.bot_logging.bot_logger.info(str(response))

            # Check if response is not None and has a 'data' key
            if response and "data" in response:
                # Convert the API response to string
                string_response = json.dumps(response["data"])
                string_response = string_response[:4000]
            else:
                string_response = "Here is a list of similar document names: "
                endpoint = f"/api/resource/{doctype}"
                endpoint = f"{endpoint}?filters=[['name','like','{docname}']]"
                endpoint = requote_uri(endpoint)
                response = _send_request("GET", endpoint)
                common.bot_logging.bot_logger.info(str(response))

                # Check if response is not None and has a 'data' key
                if response and "data" in response:
                    # Convert the API response to string
                    string_response = string_response + json.dumps(response["data"])
                else:
                    # API did not return a response or 'data' not in response
                    string_response = response

            return string_response
        except Exception as e:
            # traceback.print_exc()
            tb = traceback.format_exc()
            publish_error(e, tb)
            return tool_error(e, tb, self.description)

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("GET_CALENDAR_EVENTS does not support async")


class ERPGETFIELDS(BaseTool):
    parameters = []
    optional_parameters = []
    name = "ERP_GET_FIELDS"
    summary = """useful when you need to query a frappe/ERP next endpoint for the available fields"""
    parameters.append(
        {
            "name": "doctype",
            "description": "The doctype name that you wish to get the available fields from. enclose in double quotes",
        }
    )
    description = tool_description(name, summary, parameters, optional_parameters)
    return_direct = False

    def _run(
        self,
        doctype: str,
        publish: str = "True",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            # endpoint = f"/resource/Teams%20User"
            endpoint = f"/api/resource/DocType/{doctype}"
            endpoint = requote_uri(endpoint)
            response = _send_request("GET", endpoint)
            common.bot_logging.bot_logger.info(str(response))

            # Check if response is not None and has a 'data' key
            if response and "data" in response:
                # Convert the API response to string
                string_response = json.dumps(response["data"])
            else:
                # API did not return a response or 'data' not in response
                string_response = response

            return string_response
        except Exception as e:
            # traceback.print_exc()
            tb = traceback.format_exc()
            publish_error(e, tb)
            return tool_error(e, tb, self.description)

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("GET_CALENDAR_EVENTS does not support async")


def _send_request(method, endpoint, data=None):
    headers = {
        "Authorization": f"token {bot_config.ERP_API_KEY}:{bot_config.ERP_API_SECRET}",
        "Content-Type": "application/json",
    }
    url = f"{bot_config.ERP_URL}{endpoint}"
    url = url.replace("'", '"')
    common.bot_logging.bot_logger.info(url)
    try:
        response = requests.request(method, url, headers=headers, json=data)
        response.raise_for_status()  # Raises HTTPError if the HTTP request returned an unsuccessful status code
        return response.json()  # Returns the json-encoded content of a response, if any
    except requests.exceptions.HTTPError as http_err:
        common.bot_logging.bot_logger.error(f"HTTP error occurred: {http_err}")
        return f"HTTP error occurred: {http_err}"  # You may choose to return None or {} to signify that the request failed
    except Exception as err:
        common.bot_logging.bot_logger.error(f"An error occurred: {err}")
        return f"An error occurred: {err}"  # You may choose to return None or {} to signify that an unexpected error occurred
