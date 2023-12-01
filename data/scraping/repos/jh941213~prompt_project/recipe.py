import openai
import re

openai.api_key = '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'
model = "gpt-3.5-turbo"

def get_recipe(ingredients):

    query = f"I have {', '.join(ingredients)}. Please suggest a recipe with the title marked as #this#."
    
    messages = [{"role": "user", "content": query}]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages
    )
    answer = response['choices'][0]['message']['content']
    print(answer)
    return answer

def extract_recipe_and_menu(answer):
   
    pattern = r'\#(.*?)\#'
    recommended_dish = re.search(pattern, answer)

    if recommended_dish:
    
        dish_name = recommended_dish.group(1).replace("Recipe:", "").replace("Recipe", "").strip()
        

        parts = answer.split(recommended_dish.group(0))

        recipe_details = (parts[0] + parts[1]).replace("Title: ", "")
        
        return dish_name, recipe_details
    else:
        return None, answer
