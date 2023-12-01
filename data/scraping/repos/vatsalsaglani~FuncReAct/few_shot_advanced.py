import json
from pydantic import BaseModel

from react.func_call import FunctionCall
from config.config import OPENAI_API_KEY

class SentimentArgs(BaseModel):
    status: bool
    from_part: str

class Sentiment(BaseModel):
    is_positive: SentimentArgs
    is_exciting: SentimentArgs
    is_negative: SentimentArgs
    is_angry: SentimentArgs
    is_happy: SentimentArgs
    is_sad: SentimentArgs


SYSTEM_PROMPT = """Given a sentence in single backticks you have to classify the sentiment of the sentence into multiple labels. It can be positive, exciting, negative, angry, happy, and sad.
A sentence can have multiple labels. It is a multi-label classification problem. Along with the label the word or the combination of word that depicts that emotion should be extracted.
Example 1
Text: `I absolutely love this new update!`
Labels:
is_positive: {status: True, from_part: "love"}
is_exciting: {status: True, from_part: "absolutely love"}
is_negative: {status: False, from_part: ""}
is_angry: {status: False, from_part: ""}
is_happy: {status: True, from_part: "love"}
is_sad: {status: False, from_part: ""}

Example 2
Text: The cancellation of the event is incredibly disappointing
Labels:
is_positive: {status: False, from_part: ""}
is_exciting: {status: False, from_part: ""}
is_negative: {status: True, from_part: "disappointing"}
is_angry: {status: True, from_part: "disappointing"}
is_happy: {status: False, from_part: ""}
is_sad: {status: True, from_part: "disappointing"}
"""

print(Sentiment.schema())

fc = FunctionCall(OPENAI_API_KEY)

function_name, function_argument = fc(
    "Text: `Rainy days always make me feel a bit blue.`",
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
