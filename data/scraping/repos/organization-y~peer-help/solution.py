import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("API_KEY")

def solution_model(string):
    """ To evaluate a product's solution statement

    Args:
        string (str): section of text extracted from Notion

    Returns:
        str: GPT's evaluation of the input
    """
    try:
        response = openai.Completion.create(
            model = "text-davinci-002",
            prompt = f"The following paragraph is the solution statement of a product specification. First, evaluate and respond with a precise score from 1-100 with how well the solution statement has been written. Next, explain why this score was given along with specific feedback on what can be improved. You must give the score first and then write several in-depth sentences.\n{string}",
            temperature = 0.1,
            max_tokens = 512,
            top_p = 1,
            frequency_penalty = 0,
            presence_penalty = 0
        )
        return response["choices"][0]["text"]
    except Exception as e:
        return f"solution: {e}" # placeholder for now


# for internal testing

""" TEST_INPUT = "We introduce PEER — the Peer Editing and Efficiency Robot. PEER allows anyone to receive detailed, constructive feedback on their product spec to ensure that they are following best industry practices and that their product is off to a good start. Simply give PEER a link to your product spec as input, and PEER will give constructive feedback and suggestions on how to make sure you can make the value of your product clear, relieving you from spending the time needed to get feedback from a peer. Just go directly to the end goal with PEER. PEER will bring value to our company through its time-saving abilities by automating a crucial task generally performed by humans."

print(solution_model(TEST_INPUT)) """
