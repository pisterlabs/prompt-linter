import guidance
from config import Config
import logging

def get_food_list(ingredients, cuisine, dietrypreference, skilllevel, spicelevel, time, allergies):
    guidance.llm = guidance.llms.OpenAI("text-davinci-003", token=Config.get_openai_key())
    
    # define the prompt
    suggestions = guidance("""I need you to suggest me 10 {{ cuisine }} cuisine dishes that a person with {{ skilllevel }} cooking skill level can cook with the following ingredients : {{ ingredients }} only. within {{ time }} hours of time. I want it to be {{spicelevel}} in terms of spiciness and {{ dietrypreference }} dietary preference. also keep in mind i have {{ allergies }} allergy. I strictly need just the food names seperated with a comma.{{#geneach 'suggestions'  num_iterations=1}} {{gen 'this' temperature=0.6}} {{~/geneach}}""")
    
    # generate the list of products
    result = suggestions(ingredients = ingredients, cuisine = cuisine, dietrypreference = dietrypreference, skilllevel = skilllevel, spicelevel = spicelevel, time = time, allergies = allergies)
    print(result)
    result_string = str(result)
    print(result_string)
    new_string = result_string.split("comma.", 1)[-1]
    new_split=new_string.split(",")
    cleaned_list = [item.replace('\n', '') for item in new_split]

    return cleaned_list
