import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("API_KEY")

def schedule_model(string):
    """ To evaluate a team's planned schedule of building their product

    Args:
        string (str): section of text extracted from Notion

    Returns:
        str: GPT's evaluation of the input
    """
    try:
        response = openai.Completion.create(
            model = "text-davinci-002",
            prompt = f"The following paragraph is the schedule section of a product specification. First, evaluate and respond with a precise score from 1-100 with how well the schedule has been written. Next, explain why this score was given along with specific feedback on what can be improved. You must give the score first and then write several in-depth sentences.\n{string}",
            temperature = 0.1,
            max_tokens = 512,
            top_p = 1,
            frequency_penalty = 0,
            presence_penalty = 0
        )
        return response["choices"][0]["text"]
    except Exception as e:
        return f"schedule: {e}" # placeholder for now


# for internal testing

""" TEST_INPUT = "**Week 2:** Finalized product spec + each team members must have completed tutorials / reviewed resources and be able to speak to what is required as part of the best practices for their part in the project\
**Week 5:** Local MVP of the product\
**Week 8:** Live MVP of the product that someone else can use\
**Week 10:** Launched on [Product Hunt](https://producthunt.com/) (Note it takes 1 week after account creation to post)"

print(schedule_model(TEST_INPUT)) """
