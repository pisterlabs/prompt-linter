from decouple import config
from openai import OpenAI
from chapters import table_of_contents
import json
import pprint

client = OpenAI(api_key=config("OPENAI_API_KEY"))


def json_gpt(query, model="gpt-3.5-turbo-1106", system_message=None):
    if not system_message:
        system_message = "You are a JSON generator which outputs JSON objects according to user request"

    messages = [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": f"Please return Json for the following as instructed above:\n{query}",
        },
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )

    content: str = response.choices[0].message.content
    content: dict = json.loads(content)
    print(f"\033[94m {type(content)} \033[0m")
    pprint.pprint(content)
    return content


# json_gpt(
#     "Give me a Json object with the height in cm and age in years of all people in the following text: John is 6 feet tall and 500 months old. Mary is 5 feet tall and 30 years old. Bob is 170cm in length and was born 25 years ago."
# )

json_gpt(
    query=table_of_contents,
    system_message="""
    You are a JSON generator which outputs JSON objects according to user request.
    Please extract the author and title for all lines going all the way from start to end in the following text and return it as a JSON object following the example provided below.

    Example input:
    The Lily of Liddisdale,                         _Professor Wilson_

    The Unlucky Present,                            _Robert Chambers_

    The Sutor of Selkirk                            “_The Odd Volume_,”

    Example output:
    {'contents': [
        {'author': 'Professor Wilson', 'title': 'The Lily of Liddisdale'},
        {'author': 'Robert Chambers', 'title': 'The Unlucky Present'},
        {'author': 'The Odd Volume', 'title': 'The Sutor of Selkirk'},
    ]}
    """,
)
