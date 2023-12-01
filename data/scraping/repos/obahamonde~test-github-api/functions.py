from typing import *

import openai
from openai_function_call import openai_function, openai_schema

from .tools import Comprehend, Polly, SentimentResponse


@openai_function
def detect_sentiment(text: str) -> SentimentResponse:
    """
    Detects the sentiment of the text.

    It returns the sentiment and a confidence score for each sentiment as a percentage.
    The sentiments are: POSITIVE, NEGATIVE, NEUTRAL, MIXED.

    Args:
        text (str): The text to analyze.

    Returns:
        SentimentResponse: The sentiment and a confidence score for each sentiment as a percentage.

    Example:
        >>> detect_sentiment("I love this pizza!")
        >>> SentimentResponse(Sentiment='POSITIVE', SentimentScore={'Positive': 0.9999998807907104, 'Negative': 1.061693624787994e-7, 'Neutral': 1.061693624787994e-7, 'Mixed': 1.061693624787994e-7})
    """

    return Comprehend.from_text(text).detect_sentiment()


@openai_function
async def synthetize_speech(text: str) -> bytes:
    """
    Converts text to speech using Amazon Polly.

    It returns an ArraBuffer response to the client instead of the usual text response, it's ideal for accesibility.

    You can call this function if the user ask for an audio conversation, for example.

    Args:
        text (str): The text to convert to speech.

    Returns:
        bytes: The audio file in bytes.

    Example:
        >>> text_to_speech("Hello, how are you?")
        >>> b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00...'
    """
    polly = Polly.from_text(text)
    return await polly.get_audio()
