import openai
import os
from Utils.GPTHelper import split_into_chunks, tokenize
from dotenv import load_dotenv


class SkimifyTool():
    """
    A tool for converting dot points into more descriptive text using OpenAI GPT-3.5 Turbo.

    Attributes:
        None
    """

    def __init__(self) -> None:
        """
        Initialize the SkimifyTool instance by loading the OpenAPI API key.
        """
        load_dotenv()
        openai.api_key = os.getenv("open_ai_api_key")

    def dotpointToText(self, dotpoint: str) -> str:
        """Expand the input dot point into a more descriptive text using OpenAI GPT-3.5.

        Args:
            dotpoint (str): A dot point that needs to be expanded into a more descriptive text.

        Returns:
            str: A string containing the expanded and more descriptive version of the input dot point.
                If the tokenized input length exceeds 1500 tokens, the function returns None.

        Example:
            Input:
                "Key point 1"

            Output:
                "This is an expanded and more descriptive version of key point 1."
        """
        if len(tokenize(dotpoint)) > 1000:
            return "Input was too long, please only interact with this application though skimify.ai"

        dotpoint_conversion = ""

        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "user",
                    "content": f'Expand upon this dotpoint, give information about the topic from your own understanding but do not assume any extra details. But try to keep this relatively brief as well, three-four sentences: "{dotpoint}"'
                }
            ], temperature = 0
        )

        dotpoint_conversion += completion.choices[0].message.content
        return dotpoint_conversion


    def textToDotpoint(self, text) -> list:

        if len(text.split(" ")) < 10 and len(text) < 20:

            return "@dp Please input a larger text to Skimify!"

        if len(tokenize(text)) > 2000:
            return "@dp Please reduce input size and try again"
        

        completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f'Based solely in the information with the text, summarise the key points of this text into dotpoints labelling each dotpoint with only "@dp". The text is "{text}" '
            }
        ], temperature = 0, max_tokens = 500, n =1
    )

        data = completion.choices[0].message.content


        

        return data