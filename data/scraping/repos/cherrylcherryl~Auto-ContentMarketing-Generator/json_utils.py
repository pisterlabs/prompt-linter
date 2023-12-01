import ast
from typing import Any
import json
import os.path
from jsonschema import Draft7Validator
from typing import Tuple, Union, List

LLM_DEFAULT_RESPONSE_FORMAT = "llm_response_format"

def extract_json_from_response(response_content: str) -> dict:
    # Sometimes the response includes the JSON in a code block with ```
    if response_content.startswith("```") and response_content.endswith("```"):
        # Discard the first and last ```, then re-join in case the response naturally included ```
        response_content = "```".join(response_content.split("```")[1:-1])

    # response content comes from OpenAI as a Python `str(content_dict)`, literal_eval reverses this
    try:
        return ast.literal_eval(response_content)
    except BaseException as e:
        # TODO: How to raise an error here without causing the program to exit?
        return {}

def llm_response_schema(
    schema_name: str = LLM_DEFAULT_RESPONSE_FORMAT,
) -> dict[str, Any]:
    filename = os.path.join(os.path.dirname(__file__), f"{schema_name}.json")
    with open(filename, "r") as f:
        return json.load(f)


def validate_json(
    json_object: object, schema_name: str = LLM_DEFAULT_RESPONSE_FORMAT
) -> bool:
    """
    :type schema_name: object
    :param schema_name: str
    :type json_object: object

    Returns:
        bool: Whether the json_object is valid or not
    """
    schema = llm_response_schema(schema_name)
    validator = Draft7Validator(schema)

    if errors := sorted(validator.iter_errors(json_object), key=lambda e: e.path):
        for error in errors:
            print(f"JSON Validation Error: {error}")
        return False
    return True


def validate_market_analysis_schema(
        jdata : str
) -> Tuple[bool, Union[dict, None]] :
    try:
        jObject = json.loads(jdata)
        if "chance" not in jObject.keys():
            return False, None
        if "challenge" not in jObject.keys():
            return False, None
        if not isinstance(jObject["chance"], list):
            return False, None
        if not isinstance(jObject["challenge"], list):
            return False, None
        return True, jObject
    except:
        return False, None
    
def validate_list_schema(
        jdata : str
) -> Tuple[bool, Union[List[dict], None]] :
    try:
        jObject = json.loads(jdata)
        if not isinstance(jObject, list):
            return False, None
        for e in jObject:
            if not isinstance(e, dict):
                return False, None
            if "name" not in e.keys():
                return False, None
            if "reason" not in e.keys():
                return False, None
        return True, jObject
    except:
        return False, None
    

