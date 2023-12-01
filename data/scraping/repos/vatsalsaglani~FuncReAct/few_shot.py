import json
from pydantic import BaseModel

from react.func_call import FunctionCall
from config.config import OPENAI_API_KEY


class Sentiment(BaseModel):
    is_positive: bool
    is_exciting: bool
    is_negative: bool
    is_angry: bool
    is_happy: bool
    is_sad: bool


SYSTEM_PROMPT = """Given a sentence in single backticks you have to classify the sentiment of the sentence into multiple labels. It can be positive, exciting, negative, angry, happy, and sad.
A sentence can have multiple labels. It is a multi-label classification problem.
Example 1
Text: `I absolutely love this new update!`
Labels:
is_positive: True
is_exciting: True
is_negative: False
is_angry: False
is_happy: True
is_sad: False

Example 2
Text: The cancellation of the event is incredibly disappointing
Labels: 
is_positive: False
is_exciting: False
is_negative: True
is_angry: True
is_happy: False
is_sad: True
"""

print(Sentiment.schema())

fc = FunctionCall(OPENAI_API_KEY)

function_name, function_argument = fc(
    "Text: `I can't believe she said that to me; it's infuriating!`",
    SYSTEM_PROMPT,
    [
        {
            "name": "sentimentClassified",
            "description": "Print the categories of the sentiments of the given text.",
            "parameters": Sentiment.schema(),
        }
    ],
    "gpt-3.5-turbo",
    {"name": "sentimentClassified"},
)

print(f"FUNCTION NAME: {function_name}")
print(f"FUNCTION ARGUMENT: {json.dumps(function_argument, indent=4)}")
