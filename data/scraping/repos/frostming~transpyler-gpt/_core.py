import re
from typing import Any, Dict, Optional

import openai

SYSTEM_PROMPT = (
    "Act as a cross-version transpiler for Python language, converting Python code "
    "written in the latest version to code compatible with a specified older version "
    "of Python. The task involves replacing statements or expressions unsupported in "
    "older Python versions with their equivalent counterparts, and type annotations "
    "should be removed. I will provide the Python code snippet and the target Python "
    "version, and you will return the modified code that can run on the older Python version. "
    "Only reply with the modified code without any explanation, and don't add comments."
)

DEFAULT_OPTIONS = {
    "model": "gpt-4-1106-preview",
    "temperature": 0.0,
}

USER_PROMPT_TEMPLATE = """\
```python
{code}
```
Target Python version: {target_version}
{extras}"""

RESPONSE_REGEX = re.compile(r"```python\n(?P<code>.*)\n```", re.DOTALL)


def transpile(
    code: str,
    target_version: str,
    from_version: Optional[str] = None,
    api_key: Optional[str] = None,
    options_override: Optional[Dict[str, Any]] = None,
) -> str:
    """Transpile the given code to the target version of Python.

    Args:
        code: The code to transpile.
        target_version: The target version of Python to transpile to.
        from_version: The version of Python the code is written in. If `None`, the
            latest version of Python is assumed.

    Returns:
        The transpiled code.
    """
    client = openai.OpenAI(api_key=api_key)
    options = {**DEFAULT_OPTIONS, **(options_override or {})}
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                code=code,
                target_version=target_version,
                extras=f"From Python version: {from_version}\n" if from_version else "",
            ),
        },
    ]
    resp = client.chat.completions.create(messages=messages, **options)
    try:
        return RESPONSE_REGEX.match(resp.choices[0].message.content).group("code")
    except (AttributeError, IndexError):
        raise RuntimeError(f"Failed to transpile code, reason: {resp}") from None
