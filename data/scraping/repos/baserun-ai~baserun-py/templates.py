import hashlib
import inspect
import json
import logging
import os
import re
import traceback
from collections import defaultdict
from typing import Any, TYPE_CHECKING, Union

from Levenshtein import ratio

from baserun import Baserun
from baserun.grpc import (
    get_or_create_submission_service,
    get_or_create_async_submission_service,
)
from baserun.helpers import memoize_for_time
from baserun.v1.baserun_pb2 import (
    Template,
    TemplateVersion,
    SubmitTemplateVersionRequest,
    SubmitTemplateVersionResponse,
    GetTemplatesRequest,
    GetTemplatesResponse,
)

if TYPE_CHECKING:
    # Just for type annotations for Langchain. Since langchain is an optional dependency we have this garbage
    try:
        from langchain.tools import Tool
    except ImportError:
        Tool = Any

logger = logging.getLogger(__name__)


@memoize_for_time(os.environ.get("BASERUN_CACHE_INTERVAL", 600))
def get_templates() -> dict[str, TemplateVersion]:
    if not Baserun.templates:
        Baserun.templates = {}

    try:
        request = GetTemplatesRequest()
        response: GetTemplatesResponse = (
            get_or_create_submission_service().GetTemplates(request)
        )
        for template in response.templates:
            if template.active_version:
                version = template.active_version
            else:
                version = template.template_versions[-1]
            Baserun.templates[template.name] = version

    except BaseException as e:
        logger.error(
            f"Could not fetch templates from Baserun. Using {len(Baserun.templates.keys())} cached templates"
        )
        logger.info(traceback.format_exception(e))

    return Baserun.templates


def get_template(name: str, version: str = None) -> TemplateVersion:
    templates = get_templates()
    template = templates.get(name)
    if not template:
        logger.info(
            f"Attempted to get template {name} but no template with that name exists"
        )
        return None

    # TODO: Version lookup (instead of active version)
    return template


def most_similar_templates(formatted_str: str):
    """Given a `templates` list, will return the templates sorted by similarity to the `formatted_str` arg"""
    if not Baserun.templates:
        Baserun.templates = {}
        return []

    similarity_scores = []

    for template_version in Baserun.templates.values():
        similarity_scores.append(
            (template_version, ratio(formatted_str, template_version.template_string))
        )

    # Sort the templates by similarity ratio, in descending order
    sorted_results = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Extract only the TemplateVersion objects from the sorted list
    sorted_templates = [template for template, score in sorted_results if score > 0.5]

    return sorted_templates


def best_guess_template_parameters(
    prompt: str, template_version: TemplateVersion
) -> dict[str, Any]:
    """Given a prompt and a TemplateVersion will attempt to find the parameters that were used to generate the prompt
    Major caveats apply: this only works if the template was registered and formatted using `format_prompt` and/or
    the parameters were registered with `capture_parameters`. It's also pretty fragile because it's based on simple
    string comparison.

    The correct version of this is probably some span context propagation that I can't figure out right now.
    """
    used_parameter_list = Baserun.used_template_parameters.get(template_version.id, [])
    for used_parameters in used_parameter_list:
        # TODO: This only works for string parameters
        match = all(
            parameter in prompt
            for parameter in used_parameters.values()
            if isinstance(parameter, str)
        )
        if match:
            return used_parameters

    return {}


def is_from_jinja2_template(template_str: str, formatted_str: str):
    # Use regular expressions to extract non-variable content
    template_content = re.sub(r"{{.*?}}", "", template_str)

    # Check if all characters in template_content are present in the formatted string
    return all(char in formatted_str for char in template_content)


def extract_strings_in_braces(input_string):
    # Matches both single and double braces around the variable name
    pattern = r"\{{1,2}([^}]*)\}{1,2}"
    matches = re.findall(pattern, input_string)
    return matches


def get_template_type_enum(template_type: str = None):
    template_type = template_type or "unknown"
    if (
        template_type == Template.TEMPLATE_TYPE_JINJA2
        or template_type.lower().startswith("jinja")
    ):
        template_type_enum = Template.TEMPLATE_TYPE_JINJA2
    else:
        template_type_enum = Template.TEMPLATE_TYPE_FORMATTED_STRING

    return template_type_enum


def apply_template(
    template_string: str, parameters: dict[str, Any], template_type_enum
) -> str:
    if template_type_enum == Template.TEMPLATE_TYPE_JINJA2:
        try:
            # noinspection PyUnresolvedReferences
            from jinja2 import Template as JinjaTemplate

            template = JinjaTemplate(template_string)
            return template.render(parameters)
        except ImportError:
            logger.warning(
                "Cannot render Jinja2 template as jinja2 package is not installed"
            )
            # TODO: Is this OK? should we raise? or return blank string?
            return template_string

    return template_string.format(**parameters)


def capture_parameters(template_version_id: str, parameters: dict[str, Any]):
    if not Baserun.used_template_parameters:
        Baserun.used_template_parameters = defaultdict(list)

    Baserun.used_template_parameters[template_version_id].append(parameters)


def create_langchain_template(
    template_string: str,
    parameters: dict[str, Any] = None,
    template_name: str = None,
    template_tag: str = None,
    template_type: str = None,
    tools: list[Union["Tool", Any]] = None,
):
    from langchain.prompts import PromptTemplate

    parameters = parameters or {}
    input_variables = list(parameters.keys())
    template_type = template_type or "Formatted String"
    tools = tools or []

    if not template_name:
        caller = inspect.stack()[1].function
        template_name = f"{caller}_template"

    if (
        template_type == Template.TEMPLATE_TYPE_JINJA2
        or template_type.lower().startswith("jinja")
    ):
        langchain_template = PromptTemplate(
            template=template_string,
            input_variables=input_variables,
            template_format="jinja2",
            tools=tools,
        )

    else:
        langchain_template = PromptTemplate(
            template=template_string, input_variables=input_variables, tools=tools
        )

    template_type_enum = get_template_type_enum(template_type)
    # Support only strings for now
    parameter_definition = {var: "string" for var in input_variables}
    template_version = register_template(
        template_string=template_string,
        template_name=template_name,
        template_type=template_type_enum,
        template_tag=template_tag,
        parameter_definition=parameter_definition,
    )

    capture_parameters(template_version.id, parameters)

    return langchain_template


def format_prompt(
    template_string: str,
    parameters: dict[str, Any],
    template_name: str = None,
    template_tag: str = None,
    template_type: str = None,
    parameter_definition: dict[str, Any] = None,
):
    template_type_enum = get_template_type_enum(template_type)
    try:
        template_version = register_template(
            template_string=template_string,
            template_name=template_name,
            template_type=template_type_enum,
            template_tag=template_tag,
            parameter_definition=parameter_definition,
        )
    except BaseException as e:
        logger.warning(f"Could not register template: {e}")
        return

    capture_parameters(template_version.id, parameters)
    return apply_template(template_string, parameters, template_type_enum)


async def aformat_prompt(
    template_string: str,
    parameters: dict[str, Any],
    template_name: str = None,
    template_tag: str = None,
    template_type: str = None,
    parameter_definition: dict[str, Any] = None,
):
    template_type_enum = get_template_type_enum(template_type)
    try:
        version = await aregister_template(
            template_string=template_string,
            template_name=template_name,
            template_type=template_type_enum,
            template_tag=template_tag,
            parameter_definition=parameter_definition,
        )
    except BaseException as e:
        logger.warning(f"Could not register template: {e}")
        return

    capture_parameters(version.id, parameters)
    return apply_template(template_string, parameters, template_type_enum)


def construct_template_version(
    template_string: str,
    template_name: str = None,
    template_tag: str = None,
    template_type=Template.TEMPLATE_TYPE_FORMATTED_STRING,
    parameter_definition: dict[str, Any] = None,
) -> TemplateVersion:
    if not parameter_definition:
        parameters = extract_strings_in_braces(template_string)
        parameter_definition = {p: "string" for p in parameters}

    # Automatically generate a name based on the template's contents
    if not template_name:
        template_name = hashlib.sha256(
            f"{template_string}{parameter_definition}".encode()
        ).hexdigest()[:5]

    if not template_tag:
        template_tag = hashlib.sha256(template_string.encode()).hexdigest()[:5]

    template = Template(name=template_name, template_type=template_type)
    version = TemplateVersion(
        template=template,
        parameter_definition=json.dumps(parameter_definition),
        template_string=template_string,
        tag=template_tag,
    )

    return version


def register_template(
    template_string: str,
    template_name: str = None,
    template_tag: str = None,
    template_type=Template.TEMPLATE_TYPE_FORMATTED_STRING,
    parameter_definition: dict[str, Any] = None,
) -> TemplateVersion:
    from baserun import Baserun

    if not Baserun.templates:
        Baserun.templates = {}

    if template := Baserun.templates.get(template_name):
        return template

    version = construct_template_version(
        template_string=template_string,
        template_name=template_name,
        template_tag=template_tag,
        template_type=template_type,
        parameter_definition=parameter_definition,
    )

    request = SubmitTemplateVersionRequest(template_version=version)
    response: SubmitTemplateVersionResponse = (
        get_or_create_submission_service().SubmitTemplateVersion(request)
    )

    response_version = response.template_version
    template = response_version.template
    if template.name not in Baserun.templates:
        Baserun.templates[template.name] = response_version

    return response_version


async def aregister_template(
    template_string: str,
    template_name: str = None,
    template_tag: str = None,
    template_type=Template.TEMPLATE_TYPE_FORMATTED_STRING,
    parameter_definition: dict[str, Any] = None,
) -> TemplateVersion:
    from baserun import Baserun

    if not Baserun.templates:
        Baserun.templates = {}

    if template := Baserun.templates.get(template_name):
        return template

    version = construct_template_version(
        template_string=template_string,
        template_name=template_name,
        template_tag=template_tag,
        template_type=template_type,
        parameter_definition=parameter_definition,
    )

    request = SubmitTemplateVersionRequest(template_version=version)
    response: SubmitTemplateVersionResponse = (
        await get_or_create_async_submission_service().SubmitTemplateVersion(request)
    )

    response_version = response.template_version
    template = response_version.template
    if template.name not in Baserun.templates:
        Baserun.templates[template.name] = response_version

    return response_version
