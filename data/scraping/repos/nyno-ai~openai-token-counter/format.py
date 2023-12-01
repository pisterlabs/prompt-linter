from openai_token_counter.models import (
    ArrayProp,
    BoolProp,
    NullProp,
    NumberProp,
    ObjectProp,
    OpenAIFunction,
    PropItem,
    StringProp,
)


def format_function_definitions(functions: list[OpenAIFunction]) -> str:
    """Format the function definitions to a string.

    Args:
        functions (list[OpenAIFunction]): The list of functions to format.

    Returns:
        str: The formatted string.
    """
    lines = ["namespace functions {", ""]

    for f in functions:
        if f.description:
            lines.append(f"// {f.description}")

        parameters = f.parameters
        properties = parameters.properties if parameters else {}

        if properties is None:
            lines.append(f"type {f.name} = () => any;")
        elif len(properties) == 0:
            lines.append(f"type {f.name} = () => any;")

        else:
            lines.append(f"type {f.name} = (_: " + "{")
            lines.append(format_object_properties(f.parameters, 0))
            lines.append("}) => any;")

        lines.append("")

    lines.append("} // namespace functions")
    return "\n".join(lines)


def format_object_properties(obj: ObjectProp, indent: int) -> str:
    """Format the object properties to a string.

    Args:
        obj (ObjectProp): The object to format.
        indent (int): The indentation level.

    Returns:
        str: The formatted string.
    """
    if obj.properties is None:
        return ""

    lines = []

    required_params = obj.required or []

    for name, param in obj.properties.items():
        if param.description and indent < 2:
            lines.append(f"// {param.description}")

        if name in required_params:
            lines.append(f"{name}: {format_type(param, indent)},")
        else:
            lines.append(f"{name}?: {format_type(param, indent)},")

    return "\n".join([(" " * indent + line) for line in lines])


def format_type(param: PropItem, indent: int) -> str:
    """Format the type to a string.

    Args:
        param (PropItem): The parameter to format.
        indent (int): The indentation level.

    Returns:
        str: The formatted string.
    """
    if isinstance(param, StringProp):
        if param.enum:
            return " | ".join(f'"{v}"' for v in param.enum)  # noqa: B907
        return "string"

    if isinstance(param, NumberProp):
        if param.enum:
            return " | ".join(str(v) for v in param.enum)
        return "number"

    if isinstance(param, BoolProp):
        return "boolean"

    if isinstance(param, NullProp):
        return "null"

    if isinstance(param, ArrayProp):
        if param.items:
            return f"{format_type(param.items, indent)}[]"
        return "any[]"

    if isinstance(param, ObjectProp):
        return "{\n" + format_object_properties(param, indent + 2) + "\n}"
