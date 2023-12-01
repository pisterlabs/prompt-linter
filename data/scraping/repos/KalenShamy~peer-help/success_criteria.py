import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("API_KEY")

def success_criteria(string):
    """ To evaluate a product's success criteria
    Args:
        string (str): section of text extracted from Notion

    Returns:
        str: GPT's evaluation of the input
    """
    try:
        response = openai.Completion.create(
            model = "text-davinci-002",
            prompt = f"The following paragraph is the success criteria section of a product specification. Evaluate how well the success criteria has been described. Give specific feedback on what can be improved. Write several in-depth sentences.\n{string}",
            temperature = 1,
            max_tokens = 512,
            top_p = 0.3,
            frequency_penalty = 0,
            presence_penalty = 0
        )
        return response["choices"][0]["text"]
    except Exception as e:
        return f"solution: {e}" # placeholder for now


# for internal testing

"""TEST_INPUT = "If our product isn’t appealing to users through the feedback it provides, and they don’t find the product useful, then users won’t use it — simple as that. For that reason, the user experience is also an important aspect of our success criteria and one that we will use to make adjustments to the underlying model. In addition, another criterion for success is whether our user trusts that the model is competent at generating quality product specs. If the model were to take the feedback it gives to the user, and iterates on it’s own by editing the section it’s assessing, would the edited section be an improvement over the original section? Basically, does the model knows what it’s doing when it comes to assessing product specs?"
print(success_criteria(TEST_INPUT))"""

