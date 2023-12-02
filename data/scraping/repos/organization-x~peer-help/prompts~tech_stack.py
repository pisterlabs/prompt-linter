import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("API_KEY")

def tech_stack_model(string):
    """ To evaluate how well the product spec lists their tech stack

    Args:
        string (str): section of text extracted from Notion

    Returns:
        str: GPT's evaluation of the input
    """
    try:
        response = openai.Completion.create(
            model = "text-davinci-003",
            prompt = f"The following paragraph should be a technology stack. Evaluate the section and give specific feedback on what can be improved.\n\n\n{string}\n\n\nFEEDBACK:",
            temperature = 1,
            max_tokens = 512,
            top_p = 0.5,
            frequency_penalty = 0,
            presence_penalty = 1
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        return f"tech_stack: {e}" # placeholder for now

def suggested_tech_stack_rewrite(string, feedback):
    """ To evaluate a product's problem statement

    Args:
        string (str): section of text extracted from Notion

    Returns:
        str: GPT's evaluation of the input
    """
    try:
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=f"The following is the tech stack from a product specification and a piece of feedback assessing the quality. Rewrite the tech stack to make improvements suggested by the feedback. \n\nTECH STACK:\n\n{string}\n\nFEEDBACK:\n\n{feedback}\n\nREWRITE:\n\n",
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

"""TEST_INPUT = "- Python (Discord.py, Django)\
- OpenAI’s GPT-3 API\
- Amazon EC2"

print(tech_stack_model(TEST_INPUT))"""
