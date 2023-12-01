import openai
import logging

from django.conf import settings

logger = logging.getLogger("aichef")
def get_recipe(ingredients):
    try:
        openai.api_key = settings.OPEN_AI_KEY
        prompt = f"I have the following ingredients: {ingredients}. Kindly provide recipe for 3 meals i.e. breakfast " \
                 f"lunch and dinner ? Add title to the recipe in following format 'Breakfast:'",
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=600
        )
        return response.choices[0].text
    except openai.error.RateLimitError:
        logger.error(f"Recharge your account: {openai.error.RateLimitError.user_message}")


def get_separate_recipe(res: str):
    recipe_string = res
    # Separate recipes based on titles
    breakfast_marker = "Breakfast:"
    lunch_marker = "Lunch:"
    dinner_marker = "Dinner:"

    breakfast_start = recipe_string.find(breakfast_marker)
    lunch_start = recipe_string.find(lunch_marker)
    dinner_start = recipe_string.find(dinner_marker)

    breakfast_recipe: str = None
    lunch_recipe: str = None
    dinner_recipe: str = None

    if breakfast_start != -1:
        if lunch_start != -1:
            breakfast_recipe = recipe_string[breakfast_start:lunch_start].strip()
        elif dinner_start != -1:
            breakfast_recipe = recipe_string[breakfast_start:dinner_start].strip()
        else:
            breakfast_recipe = recipe_string[breakfast_start:].strip()

    if lunch_start != -1:
        if dinner_start != -1:
            lunch_recipe = recipe_string[lunch_start:dinner_start].strip()
        else:
            lunch_recipe = recipe_string[lunch_start:].strip()

    if dinner_start != -1:
        dinner_recipe = recipe_string[dinner_start:].strip()
    return {
        "breakfast": breakfast_recipe,
        "lunch": lunch_recipe,
        "dinner": dinner_recipe
    }