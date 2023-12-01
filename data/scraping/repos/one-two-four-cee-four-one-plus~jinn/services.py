import json
import inspect
import traceback

try:
    from openai import OpenAI
except ImportError:
    import pip
    pip.main(['install', 'openai==1.3.5'])
    from openai import OpenAI

from constants import OPENAI_FUNCTION_SCHEMA, CRAFT_INCANTATION_SCHEMA
from utils import unwrap_content, define_function, NoDefaults


def describe_function(key, model, code):
    response = OpenAI(api_key=key).chat.completions.create(
        model=model,
        temperature=0,
        messages=[{
            "role": "user",
            "content": (
                f"Describe this python function: {code} in this schema {OPENAI_FUNCTION_SCHEMA}. "
                "I want only json in response, nothing else."
            )
        }]
    )
    return unwrap_content(response.choices[0].message.content, 'json')


def craft_incantation(key, model, retries, text):
    messages = [{
        "role": "user",
        "content": (
            "Write a python function according to the request.  I want only"
            " python code in response, nothing else. Don't comment the code,"
            " don't ask for user's input, don't print anything and don't do any"
            " exception handling unless it's necessary. Don't use keyword,"
            " variadic arguments and types other than int, float, str, bool as"
            " function's arguments. Don't use any global variables. If you need"
            " any external libraries, try to import them within function's body"
            " with try/except block. If said libraries are not part of standard library"
            " and are not available, import pip module and install them. Example:\n"
            "try:\n"
            "    import numpy\n"
            "except ImportError:\n"
            "    import pip\n"
            "    pip.main(['install', 'numpy'])\n"
            "If function needs any parameters (like credentials,"
            " configuration, etc), they should be passed as arguments. \nRequest:"
            f"{text}"
        )
    }]
    response = OpenAI(api_key=key).chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    messages.append({"role": "assistant", "content": response.choices[0].message.content})

    last_e = None
    for i in range(int(retries)):
        code = unwrap_content(response.choices[0].message.content, 'python')
        code = NoDefaults.in_(code)
        try:
            name, _ = define_function(code)
            return name, code
        except Exception as e:
            last_e = e
            messages.append({
                "role": "assistant",
                "content": (
                    f"{''.join(traceback.format_exception(e))}\n"
                    "Fix this error. I want only python code in response, nothing else."
                )
            })
            continue
    else:
        return last_e


def wish(key, model, text, incantations, allow_craft=False):
    tools = [value['schema'] for value in incantations.values()]
    instructions = ' Use external tool.'
    if allow_craft:
        tools.append(CRAFT_INCANTATION_SCHEMA)
        instructions += (
            ' If there is no suitable tool available, define one instead of fulfilling'
            ' the original request. The tool should be generic enough to be useful in'
            ' other situations.'
        )
    response = OpenAI(api_key=key).chat.completions.create(
        model=model,
        temperature=0,
        messages=[{
            "role": "user",
            "content": text + instructions
        }],
        tools=tools,
        tool_choice="auto"
    )
    response_message = response.choices[0].message
    if tool_calls := response_message.tool_calls:
        if tool_calls[0].function.name == 'craft_incantation':
            tool_text = json.loads(tool_calls[0].function.arguments)['text']
            return tool_text

        incantation = incantations[tool_calls[0].function.name]
        _, func = define_function(incantation['code'])
        args = json.loads(tool_calls[0].function.arguments)
        try:
            return func(**args)
        except Exception as e:
            return incantation['object'], tool_calls[0].function.arguments, e
    else:
        return response_message.content


def fix(key, model, code, request, traceback):
    _, func = define_function(code)
    arguments = set(inspect.getargspec(func).args)
    arguments = {k: v for k, v in json.loads(request).items() if k in arguments}
    arguments = json.dumps(arguments)
    response = OpenAI(api_key=key).chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": (
            "I need you to fix python function. I will provide it's code, call"
            " arguments formatted in some json schema and error traceback. Fix"
            " this error. I want only python code in response, nothing else."
            f" Code:\n{code}\nArguments:\n{arguments}\n"
            f"Traceback:\n{traceback}"
        )}]
    )
    code = unwrap_content(response.choices[0].message.content, 'python')
    try:
        define_function(code)
        return code
    except Exception as e:
        return e
