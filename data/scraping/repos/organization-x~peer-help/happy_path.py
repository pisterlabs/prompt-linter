import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("API_KEY")

def happy_path_model(string):
    """ To evaluate a product's intended happy path

    Args:
        string (str): section of text extracted from Notion

    Returns:
        str: GPT's evaluation of the text
    """
    try:
        response = openai.Completion.create(
            model = "text-davinci-002",
            prompt = f"The following paragraph is the happy path section of a product specification. Evaluate the happy path and give specific feedback on what can be improved.\n\n\n{string}\n\n\nFEEDBACK:",
            temperature = 0.5,
            max_tokens = 512,
            top_p = 1,
            frequency_penalty = 0,
            presence_penalty = 0
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        return f"happy_path: {e}" # placeholder for now

def suggested_happy_path_rewrite(string, feedback):
    """ To evaluate a product's problem statement

    Args:
        string (str): section of text extracted from Notion

    Returns:
        str: GPT's evaluation of the input
    """
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"The following is the happy path from a product specification and a piece of feedback assessing the quality. Rewrite the happy path to make improvements suggested by the feedback. \n\nHAPPY PATH:\n\n{string}\n\nFEEDBACK:\n\n{feedback}\n\nREWRITE:\n\n",
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0.95,
            presence_penalty=0.95
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        return f"problem: {e}" # placeholder for now

# for internal testing

"""TEST_INPUT = "1. Prompt-engineering GPT-3 to numerically evaluate sections to determine the prompt that is given to the GPT-3 model for feedback generation.\
2. Prompt-engineering GPT-3 to generate feedback based on numerical scores given by the scoring method.\
3. Extracting and parsing text from Notion product specs into individual sections (problem statement, solution, etc.)\
4. Backend to feed sections to the scoring method to the GPT-3 model using a bot command and generating an embed."

print(happy_path_model(TEST_INPUT))"""
