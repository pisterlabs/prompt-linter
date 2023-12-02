import openai
import os

openai.api_key = os.environ["OPENAI_API_KEY"]

VALIDATION_MESSAGE = "Your job is to see if the following output is valid for the given pseudocode and (potentially) template. If it is valid, return 'Valid'. If it is not valid, return 'Invalid' and explain why it is invalid."

RULES = [
    "The output code should follow the template code but also match the pseudocode.",
    "The output code should be syntactically valid.",
]


def validate_output(pseudo, template, output):
    system_msg = VALIDATION_MESSAGE + " ".join(RULES)
    usr_msg = f"Pseudocode: {pseudo}\nTemplate: {template}\nOutput: {output}"

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

    return response.choices[0].message.content
