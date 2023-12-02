from enum import Enum
import json
from pydantic import BaseModel

from react.func_call import FunctionCall
from config.config import OPENAI_API_KEY


class Sentiments(Enum):
    positive = "POSITIVE"
    negative = "NEGATIVE"
    neutral = "NEUTRAL"


class Sentiment(BaseModel):
    category: Sentiments


SYSTEM_PROMPT = """Given a sentence in single backticks you have to classify the sentiment of the sentence into one of the following categories, POSITIVE, NEGATIVE, NEUTRAL"""

print(Sentiment.schema())

fc = FunctionCall(OPENAI_API_KEY)

function_name, function_argument = fc(
    "Text: `The movie was great!`",
    SYSTEM_PROMPT,
    [
        {
            "name": "sentimentClassified",
            "description": "Print the category of the sentiment of the given text.",
            "parameters": Sentiment.schema(),
        }
    ],
    "gpt-3.5-turbo",
    {"name": "sentimentClassified"},
)

print(f"FUNCTION NAME: {function_name}")
print(f"FUNCTION ARGUMENT: {json.dumps(function_argument, indent=4)}")
