import os
import openai
from typing import Dict, Optional

openai.api_key = "YOU_API_KEY"


def eval_fun(inputs: Dict[str, str], output: str, target: Optional[str] = None) -> float:
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Evaluate if an answer contains a thought process or is only providing an answer without "
                           "explanation. Say 'thoughts' if there was a thought process. Say 'only answer' if it was "
                           "only an answer."
            },
            {"role": "user", "content": output}
        ],
    )
    response_text = completion.choices[0].message["content"]
    return float('thoughts' in response_text.lower())
