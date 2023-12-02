import openai
import os

import pseudo.config as config

VALIDATION_MESSAGE = "Your job is to find any errors between the pseudocode, output, and template code. Explain why the output code is invalid."
FIX_MESSAGE = "Using the given comments, fix the code so that it is valid."

RULES = [
    # "The output code should follow the template code but also match the pseudocode.",
    "The output code should be syntactically valid.",
    "Check if the output code is missing any classes or functions from the pseudocode.",
    "Check if the output code contains any undefined variables or functions.",
    "If the code is actually valid, return 'Valid'",
]


def validate_output(pseudo, template, output):
    system_msg = VALIDATION_MESSAGE + " ".join(RULES)
    # usr_msg = f"Pseudocode: {pseudo}\nTemplate: {template}\nOutput: {output}"
    usr_msg = f"<Pseudocode>{pseudo}</Pseudocode><Output>{output}</Output>"

    msg = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": usr_msg},
    ]

    # Use the OpenAI API to generate code from the file contents
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=msg,
        temperature=0.5,
    )

    validation_output = response.choices[0].message.content

    if config.LOG_MSG:
        print("Validation output: ")
        print(validation_output)

    return validation_output


def fix_code(code, validation_message):
    system_msg = FIX_MESSAGE
    usr_msg = f"<Code>{code}</Code><Fixes>{validation_message}</Fixes>"

    msg = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": usr_msg},
    ]

    # Use the OpenAI API to generate code from the file contents
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=msg,
        temperature=0.5,
    )

    output = response.choices[0].message.content
    if config.LOG_MSG:
        print("Message: ")
        print(msg)

        print("Fix output: ")
        print(output)

    return output
