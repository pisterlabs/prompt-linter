from openai import OpenAI


def reminder_example():
    f = make_funcall_tool(
        "make_reminder",
        "make reminder with time range, like 12h, 1d, 30m (only accept s, m, h, d); or timestamp only has hours (24) and min like 14:34",
        {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "the content we are trying to reminder",
                },
                "timestamp": {"type": "string", "description": "loop range or specific timestamp (hours:mins)"},
            },
        }
    )
    return f


def make_funcall_tool(fn_name, desc, parameters):
    f = {
        "type": "function",
        "function": {
            "name": fn_name,
            "description": desc,
            "parameters": parameters
        }}

    return f
