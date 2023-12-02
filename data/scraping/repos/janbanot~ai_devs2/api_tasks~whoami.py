import os
import json
import openai
from dotenv import load_dotenv
from ai_devs_task import Task
from typing import Dict, Any, List, Tuple

# TODO add typing

load_dotenv()
ai_devs_api_key: str = os.getenv("AI_DEVS_API_KEY", "")
openai.api_key = os.getenv("OPENAI_API_KEY", "")

whoami: Task = Task(ai_devs_api_key, "whoami")

context: List[str] = []
prompt: str = """
You are trying to guess the name of the person based on the hints given below.
Answer shortly with name and surname the question only if you are certain.
If you are not certain answer with "HINT"

### Hints:
"""


def guess(task: Task, context: List[str], prompt: str) -> Tuple[str, str]:
    token: str = task.auth()
    content: Dict[str, Any] = whoami.get_content(token)
    hint: str = content["hint"]
    context.append(hint)
    enriched_prompt: str = enrich_prompt(prompt, context)
    check_answer: openai.ChatCompletion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": enriched_prompt},
            {"role": "user", "content": "Who am I?"}
        ]
    )
    return (token, check_answer["choices"][0]["message"]["content"])


def enrich_prompt(prompt: str, context: List[str]) -> str:
    for hint in context:
        prompt += f"- {hint}\n"
    return prompt


def function_calling(query: str) -> Dict[str, Any]:
    function_descriptions: List[Dict[str, Any]] = [
                {
                    "name": "post_answer",
                    "description": "If input is a name post answer",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Guessed name of the person",
                            }
                        },
                        "required": ["name"]
                    },
                },
                {
                    "name": "ask_for_hint",
                    "description": "If input is 'HINT' ask for another hint",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "hint": {
                                "type": "string",
                                "description": "",
                            }
                        }
                    },
                }
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[{"role": "user", "content": query}],
        functions=function_descriptions
    )
    response_message = response["choices"][0]["message"]

    return response_message["function_call"]


def solve():
    response = guess(whoami, context, prompt)
    token, guessed_answer = response
    response = function_calling(guessed_answer)
    return (token, response)


while True:
    token, function_call = solve()
    if function_call["name"] == "post_answer":
        name = json.loads(function_call["arguments"])["name"]
        answer_payload: Dict[str, str] = {"answer": name}
        response = whoami.post_answer(token, answer_payload={"answer": function_call["arguments"]})
        print(response)
        break
    elif function_call["name"] == "ask_for_hint":
        token, function_call = solve()
    else:
        print("Something went wrong")
        break
