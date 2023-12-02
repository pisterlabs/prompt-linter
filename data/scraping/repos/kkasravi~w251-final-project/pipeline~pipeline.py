import os
import openai
import sys
import re

openai.api_key = os.getenv("OPENAI_API_KEY")
MODELS = ["gpt-3.5-turbo","gpt-4","text-davinci-003","davinci-instruct-beta:2.0.0"]
SYSTEM_INPUT = '''
You are an expert in cooking and cuisine. You are instructed to do the following:

1. Take one or more ingredients as input.
2. Take a cuisine as input.
3. Produce a recipe for that cuisine that includes the ingredients and ingredient amounts.
4. Produce detailed cooking instructions.
5. Produce a description of the meal produced by the recipe.
'''

USER_INPUT1 = '''
Ingredients: chicken breast, onions, garlic, bell pepper
Cuisine: Mexican
'''

ASSISTANT_INPUT = '''
Generate a Mexican recipe that includes chicken breast, onions, garlic, and bell pepper as the main ingredients. The recipe should include ingredient amounts and detailed cooking instructions. The final dish should be described in a way that can be used to generate an image.
'''

def parse_input(input_str):
    ingredients_pattern = r'Ingredients:\s*(.*)'
    cuisine_pattern = r'Cuisine:\s*(.*)'
    
    ingredients_match = re.search(ingredients_pattern, input_str, re.IGNORECASE)
    cuisine_match = re.search(cuisine_pattern, input_str, re.IGNORECASE)
    
    if not ingredients_match or not cuisine_match:
        raise ValueError("Invalid input format. Expected format: 'Ingredients: <comma-separated ingredients>\\nCuisine: <cuisine>'")
    
    ingredients_str = ingredients_match.group(1)
    cuisine = cuisine_match.group(1).strip()
    
    ingredients = [ingredient.strip() for ingredient in ingredients_str.split(',')]
    
    return ingredients, cuisine


def parse_description(input_string):
    description_start = input_string.find("Description:")
    
    if description_start == -1:
        return "Description not found."
    
    description = input_string[description_start + len("Description:"):].strip()
    return description


def generate_recipe(ingredients, cuisine):
    ingredients = "Ingredients: " + ', '.join(ingredients)
    cuisine = "Cuisine: " + cuisine
    user_input = '\n'.join([ingredients, cuisine])
    assistant_input = "Title: Mexican Chicken and Poblano Pepper Casserole Recipe\n\nIntro: \nThis Mexican-inspired casserole is the perfect dish for a cozy night in. With tender chicken breast, sweet onion, and spicy poblano peppers, this dish is sure to satisfy your cravings. Follow these simple steps to make this delicious meal!\n\nIngredients:\n- 2 chicken breasts\n- 1 large onion, chopped\n- 2 poblano peppers, seeded and chopped\n- 2 tsp. olive oil\n- 1 tsp. ground cumin\n- 1 tsp. chili powder\n- 1/2 tsp. salt\n- 1/2 tsp. black pepper\n- 1/2 cup chicken broth\n- 1 cup cooked rice\n- 1/2 cup shredded cheddar cheese\n\nInstructions:\n1. Preheat the oven to 375 degrees F.\n2. Heat the olive oil in a large skillet over medium-high heat. Add the chicken breasts and cook until browned on both sides, about 4-5 minutes per side. Remove the chicken from the skillet and set aside.\n3. Add the onion and poblano peppers to the same skillet and cook until the vegetables are softened, about 5 minutes.\n4. Stir in the cumin, chili powder, salt, and black pepper. Cook for an additional minute.\n5. Add the chicken broth to the skillet and stir to combine everything well.\n6. Transfer the chicken and vegetable mixture to a baking dish.\n7. Cover the baking dish with foil and bake for 25 minutes.\n8. Remove the foil and sprinkle the shredded cheese over the top of the casserole.\n9. Bake uncovered for an additional 10-15 minutes, or until the cheese is melted and bubbly.\n10. Serve hot over cooked rice.\n\nDescription:\nThis Mexican-inspired casserole is the perfect mix of bold flavors and comforting ingredients. Tender chicken breast, sweet onion, and spicy poblano peppers come together in a creamy, cheesy sauce that's sure to satisfy any craving. Served over a bed of white rice, this dish is the ultimate comfort food for any night of the week. Enjoy!"

    response = openai.ChatCompletion.create(
      model=MODELS[0],
      messages=[
            {"role": "system", "content": SYSTEM_INPUT},
            {"role": "user", "content": USER_INPUT1, "name": "example_user"},
            {"role": "assistant", "content": assistant_input, "name": "example_assistant"},
            {"role": "user", "content": user_input}
        ]
    )

    return response


def generate_image(description):
    response = openai.Image.create(
        prompt=description,
        n=1,
        size="1024x1024"
    )

    image_url = response['data'][0]['url']
    return image_url


if __name__ == "__main__":
    input_str = sys.stdin.read()
    ingredients, cuisine = parse_input(input_str)
    response = generate_recipe(ingredients, cuisine)
    response = response['choices'][0]['message']['content']
    description = parse_description(response)
    image_url = generate_image(description)

    print(response)
    print('\n\n')
    print(image_url)
