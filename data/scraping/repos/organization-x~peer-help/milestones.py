import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("API_KEY")

def milestones_model(string):
    """ To evaluate a product's milestones

    Args:
        string (str): section of text extracted from Notion

    Returns:
        str: GPT's evaluation of the input
    """
    try:
        response = openai.Completion.create(
            model = "text-davinci-003",
            prompt = f"The following paragraph is the milestones section of a product specification. Evaluate the milestones and give specific feedback on what can be improved.\n\n\n{string}\n\n\nFEEDBACK:",
            temperature = 1,
            max_tokens = 512,
            top_p = 0.5,
            frequency_penalty = 0,
            presence_penalty = 0,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        return f"milestones: {e}" # placeholder for now

def suggested_milestones_rewrite(string, feedback):
    """ To evaluate a product's problem statement

    Args:
        string (str): section of text extracted from Notion

    Returns:
        str: GPT's evaluation of the input
    """
    try:
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=f"The following are the milestones from a product specification and a piece of feedback assessing the quality. Rewrite the milestones to make improvements suggested by the feedback. \n\nMILESTONES:\n\n{string}\n\nFEEDBACK:\n\n{feedback}\n\nREWRITE:\n\n",
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

"""TEST_INPUT = "- ML:\
    - Automating prompt-engineering the GPT-3 API to generate valuable feedback.\
    - Developing a scoring method to assign each section a score to determine the prompt that will be given to the GPT-3 model.\
    - Testing and Evaluation of Generated Output.\
- Backend:\
    - Engineering the Notion API to extract text from product specs.\
    - Parsing the product spec into individual sections (e.g., problem statement, solution).\
    - Pipeline That Passes Text to the Model\
    - Testing and Deployment\
- Frontend:\
    - Bot Development and Output Display\
    - Testing and Deployment\
\
In addition, our milestones also include ensuring that our success criteria is being met by monitoring and prioritizing our success metrics - or at least the metrics we can monitor for now (linter score, amount of testing code)."

print(milestones_model(TEST_INPUT))"""
