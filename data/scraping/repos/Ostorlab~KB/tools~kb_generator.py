"""Module responsible for interacting with the OpenAI API to generate KB entries."""
import dataclasses
import enum
import json
import logging
import os
import pathlib
import re
from typing import Any

import click
import openai
import tenacity
from openai import openai_object
from openai.api_resources import chat_completion

DEMO_LANGS = ["Flutter", "Swift", "Kotlin"]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-3.5-turbo"

DESCRIPTION_TEMPLATE = """
# %%VULNERABILITY_NAME%%

%%VULNERABILITY_DESCRIPTION%%

### Examples

#### Dart

```dart
%%FLUTTER_CODE%%
```

#### Swift

```swift
%%SWIFT_CODE%%
```

#### Kotlin

```kotlin
%%KOTLIN_CODE%%
```
"""


RECOMMENDATION_TEMPLATE = """
# %%VULNERABILITY_NAME%%

%%RECOMMENDATION%%

# Code Examples:

### Dart

```dart
%%FLUTTER_CODE%%
```

### Swift

```swift
%%SWIFT_CODE%%
```

### Kotlin

```kotlin
%%KOTLIN_CODE%%
```
"""

META_TEMPLATE = """
{
   "risk_rating":"[hardening/info/low/medium/high]",
   "short_description":"[short description of the vulnerability]",
   "references":{
      "Vulnerability (SOURCE)":"[SOURCE LINK]",
      ...
   },
   "title":"[Vulnerability title]",
   "privacy_issue":[true/false],
   "security_issue":[true/false],
   "categories":{
       "OWASP_MASVS_L1":[
         "MSTG reference" 
      ],
      "OWASP_MASVS_L2":[
         "MSTG reference"
      ]
   }
}
"""

PATTERN = "```(.*)```"


@dataclasses.dataclass
class RiskRating(enum.Enum):
    """RiskRating dataclass"""

    INFO = "INFO"
    HARDENING = "HARDENING"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclasses.dataclass
class Platform(enum.Enum):
    """Platform dataclass"""

    IOS = "IOS"
    ANDROID = "ANDROID"
    MULTI = "MULTIPLATFORM"
    COMMON = "COMMON"
    WEB = "WEB"


@dataclasses.dataclass
class Vulnerability:
    """Vulnerability dataclass"""

    name: str
    risk_rating: RiskRating
    platform: Platform


@dataclasses.dataclass
class KBEntry:
    """KBEntry dataclass"""

    description: str
    recommendation: str
    meta: dict[str, Any]
    vulnerability: Vulnerability


PLATFORM_TO_PATH = {
    "IOS": "MOBILE_CLIENT/IOS/",
    "ANDROID": "MOBILE_CLIENT/ANDROID/",
    "COMMON": "MOBILE_CLIENT/COMMON/",
    "MULTIPLATFORM": "MOBILE_CLIENT/MULTIPLATFORM_JS/",
    "WEB": "WEB_SERVICE/WEB/",
}


def dump_kb(kbentry: KBEntry) -> None:
    """Dump KB entry into files.

    Args:
        kbentry: Knowledge entry object.
    Returns:

    """
    path_prefix = pathlib.Path(
        PLATFORM_TO_PATH[kbentry.vulnerability.platform.value],
        f"_{kbentry.vulnerability.risk_rating.value}",
    )

    logging.info("KB generated successfully, path is %s", path_prefix)

    path_prefix.mkdir(exist_ok=True, parents=True)

    with pathlib.Path(path_prefix, "description.md").open(
        "w", encoding="utf-8"
    ) as description_md:
        description_md.write(kbentry.description)

    with pathlib.Path(path_prefix, "recommendation.md").open(
        "w", encoding="utf-8"
    ) as recommendation_md:
        recommendation_md.write(kbentry.recommendation)

    with pathlib.Path(path_prefix, "meta.json").open(
        "w", encoding="utf-8"
    ) as meta_json:
        json.dump(kbentry.meta, meta_json, indent=4)


def _ask_gpt(
    prompts: list[dict[str, str]], temperature: float = 0.0, max_tokens: int = 3200
) -> openai_object.OpenAIObject:
    """Send a prompt to OpenAI API."""
    if OPENAI_API_KEY is None:
        raise ValueError

    openai.api_key = OPENAI_API_KEY
    gpt_response: openai_object.OpenAIObject = chat_completion.ChatCompletion.create(
        model=MODEL_NAME,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=prompts,
    )
    return gpt_response


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_fixed(2),
    retry=tenacity.retry_if_exception_type(),
)
def generate_kb(vulnerability: Vulnerability) -> KBEntry:
    """Send a prompt to the OpenAI API and generate KB.

    Args:
        vulnerability: a vulnerability object
    Returns:
        KB entry

    """
    description_md = DESCRIPTION_TEMPLATE.replace(
        "%%VULNERABILITY_NAME%%", vulnerability.name.title()
    )
    prompt_message = (
        f"Vulnerability description for {vulnerability.name}, reply as one short paragraph without "
        f"mitigation details "
    )
    prompts = [
        {
            "role": "user",
            "content": prompt_message,
        },
    ]
    gpt_response = _ask_gpt(prompts=prompts)
    content = gpt_response.choices[0].message["content"]
    description_md = description_md.replace("%%VULNERABILITY_DESCRIPTION%%", content)

    recommendation_md = RECOMMENDATION_TEMPLATE.replace(
        "%%VULNERABILITY_NAME%%", vulnerability.name.title()
    )
    prompt_message = f"Vulnerability mitigation for {vulnerability.name}, reply as one short paragraph"
    prompts = [
        {
            "role": "user",
            "content": prompt_message,
        },
    ]
    gpt_response = _ask_gpt(prompts=prompts)
    content = gpt_response.choices[0].message["content"]
    recommendation_md = recommendation_md.replace("%%RECOMMENDATION%%", content)

    for language in DEMO_LANGS:
        prompt_message = (
            f"Demo {language} application that is vulnerable to {vulnerability.name}, "
            "vulnerability has to depend on user input, "
            "application code has to include imports and main function"
            "your response need to consist of the code alone without any extra text"
        )
        prompts = [
            {
                "role": "system",
                "content": "act as a code generator, only reply with code, nothing else",
            },
            {
                "role": "user",
                "content": prompt_message,
            },
        ]
        gpt_response = _ask_gpt(prompts=prompts)
        content = gpt_response.choices[0].message["content"]
        match = re.search(PATTERN, content, re.DOTALL | re.MULTILINE)
        code = match.group(1).strip() if match else "[TODO]"
        description_md = description_md.replace(f"%%{language.upper()}_CODE%%", code)

        prompt_message = (
            f"The {language} code below is vulnerable to {vulnerability.name}, generate a patched version of it:"
            f"{content}"
        )
        prompts = [
            {
                "role": "system",
                "content": "act as a code generator, only reply with code, nothing else",
            },
            {
                "role": "user",
                "content": prompt_message,
            },
        ]
        gpt_response = _ask_gpt(prompts=prompts)
        content = gpt_response.choices[0].message["content"]
        match = re.search(PATTERN, content, re.DOTALL | re.MULTILINE)
        code = match.group(1).strip() if match else "[TODO]"
        recommendation_md = recommendation_md.replace(
            f"%%{language.upper()}_CODE%%", code
        )

    prompt_message = (
        f"Generate a metadata for {vulnerability.name} vulnerability, use the following template: "
        f"{META_TEMPLATE}"
    )
    prompts = [
        {
            "role": "system",
            "content": "act as a json metadata generator, only reply with json, nothing else",
        },
        {
            "role": "user",
            "content": prompt_message,
        },
    ]
    gpt_response = _ask_gpt(prompts=prompts)
    content = gpt_response.choices[0].message["content"]
    meta = json.loads(content)

    kbentry = KBEntry(description_md, recommendation_md, meta, vulnerability)

    return kbentry


@click.command()
@click.option("--name", prompt="Enter vulnerability name", help="Vulnerability name")
@click.option(
    "--risk",
    prompt="Enter risk rating",
    help="Risk rating",
    type=click.Choice([risk.value for risk in RiskRating], case_sensitive=False),
)
@click.option(
    "--platform",
    prompt="Enter platform",
    help="Platform",
    type=click.Choice([platform.value for platform in Platform], case_sensitive=False),
)
def main(name: str, risk: str, platform: str) -> None:
    """
    Entry point of the program.

    This function executes the main logic of the program, including initialization,
    user interactions, and finalization steps. It serves as the starting point for
    running the application.

    Args:
        name: vulnerability name
        risk: vulnerability risk rating
        platform: vulnerability target platform
    Returns:

    """
    vulnerability = Vulnerability(name, RiskRating(risk), Platform(platform))
    kbentry = generate_kb(vulnerability)
    dump_kb(kbentry)


if __name__ == "__main__":
    main()
