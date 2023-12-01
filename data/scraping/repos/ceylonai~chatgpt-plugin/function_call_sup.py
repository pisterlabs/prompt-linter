from typing import Any, Dict, Optional

from prance import ResolvingParser
from pydantic.fields import Undefined
import inspect

from dotenv import load_dotenv

load_dotenv(".env", )


def Query(  # noqa: N802
        default: Any = Undefined,
        *,
        alias: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        required: bool = False,
        gt: Optional[float] = None,
        ge: Optional[float] = None,
        lt: Optional[float] = None,
        le: Optional[float] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        regex: Optional[str] = None,
        example: Any = Undefined,
        examples: Optional[Dict[str, Any]] = None,
        deprecated: Optional[bool] = None,
        include_in_schema: bool = True,
        **extra: Any,
) -> Any:
    class QueryCls:
        def __init__(self, default: Any = Undefined, **kwargs):
            self.default = default
            self.__dict__.update(kwargs)

    return QueryCls(default,
                    alias=alias,
                    title=title,
                    required=required,
                    description=description,
                    gt=gt,
                    ge=ge,
                    lt=lt,
                    le=le,
                    min_length=min_length,
                    max_length=max_length,
                    regex=regex,
                    example=example,
                    examples=examples,
                    deprecated=deprecated,
                    include_in_schema=include_in_schema,
                    **extra)


def get_function_detail(func):
    _name = func.__name__
    _description = func.__doc__
    _detail = func.__defaults__
    inspect.getfullargspec(func)

    properties = {}
    required = []
    sig = inspect.signature(func)
    params = sig.parameters
    for name, param in params.items():
        query = param.default
        _type = type(query.default)

        if _type == str:
            _type = "string"
        elif _type == int:
            _type = "integer"
        elif _type == bool:
            _type = "boolean"
        elif _type == float:
            _type = "number"

        properties[name] = {
            "type": _type,
            "description": query.description,
        }

        if query.required:
            required.append(name)

    func_detail = {
        "name": _name,
        "description": _description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }
    return func_detail


def reg_functions(funcs: []):
    _functions = []
    _function_calls = {}
    for func in funcs:
        func_detail = get_function_detail(func)
        _functions.append(func_detail)
        _function_calls[func_detail["name"]] = func
    return _functions, _function_calls


def extract_api(api_schema_url):
    parser = ResolvingParser(api_schema_url)

    servers = parser.specification.get('servers', [])
    server_url = None
    for server in servers:
        server_url = server["url"]

    paths = parser.specification['paths']

    actions = []
    calling_data = {

    }

    for path, methods in paths.items():
        # print(f'Path: {BASE_ULR}{path}')
        for method, operation in methods.items():
            name = operation.get("summary", "No summary provided")
            _description = operation.get("description", "No description provided")

            params = operation.get("parameters", [])
            required = []
            properties = {}
            for param in params:
                param_name = param.get("name")
                _type = param.get("schema", {}).get("type")
                _description = param.get("description")

                if _type == "array":
                    _type = "string"

                properties[param_name] = {
                    "type": _type,
                    "description": _description
                }
                if param.get("required"):
                    required.append(param_name)

            tool_name = f"{name}_{method}".replace(" ", "_").lower()
            func_detail = {
                "name": tool_name,
                "description": _description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            }
            actions.append(func_detail)

            calling_data[tool_name] = {
                "path": f"{server_url}{path}",
                "method": method
            }
            # break
        # break
    return actions, calling_data


def process_conversation(messages,
                         function_details,
                         function_calls,
                         auth_func=None,
                         model="gpt-3.5-turbo-0613"):
    import os
    import openai
    import json
    import requests
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        functions=function_details,
        function_call="auto",  # auto is default, but we'll be explicit
    )
    response_message = response["choices"][0]["message"]
    # Step 2: check if GPT wanted to call a function
    if response_message.get("function_call"):
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        # available_functions = function_calls
        function_name = response_message["function_call"]["name"]
        function_to_call = function_calls[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        print(function_args, function_to_call)

        # Convert params to JSON string
        params_json = function_args
        url = function_to_call["path"]

        headers = {}

        if auth_func:
            headers["Authorization"] = auth_func()
        # Send the request
        response = requests.get(url, headers=headers, params=params_json)

        # Check if the request was successful
        if response.status_code == 200:
            # Do something with the response
            print(response.json())
        else:
            print(f'Request failed with status code {response.status_code}')


BASE_ULR = "https://<plugin>.ceylon.ai"i"
OPEN_API_URL = f"{BASE_ULR}/openapi.json"

actions, calling_data = extract_api(OPEN_API_URL)

# print(json.dumps(actions, indent=4))
# pprint(actions)
_messages = [{"role": "user", "content": "Generate photo realistic images of people working in an office"}]


def auth_func():
    """
    This function should return the auth token
    :return:
    """
    pass


res = process_conversation(_messages, actions, calling_data, auth_func=auth_func)
print(res)
