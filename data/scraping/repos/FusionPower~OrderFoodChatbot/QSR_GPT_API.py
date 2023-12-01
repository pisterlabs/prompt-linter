"""
The chatbot can take several actions that are numbered to classify them.
Some of the actions have specifications on what the expected completion
should be formated as (e.g. price(item)).

Chatbot actions:
0 regular conversation
1 greet

CRUD:
2 add items to the order
3 read an order
4 update an item
5 delete an item

6 Menu Lookups: price(item), ingredients(item), vegetarian(*)
  price
  ingredients
  vegetarian optons *
  gluten free *

  
7 Subitem configuration (e.g., extra pickles, size, etc.) TODO
    //hamburger +pickels, -tomatoes, +cheese, +large


8 affirmative, confirmative
    - if 2,3,4 in previous actions, ask if they want anything else
    - if 8 in previous actions, ask what else they want
9 negative
    - if 2 in previous actions, undo changes
    - if 8 in previous actions, end conversation 

TODO Possible expansions:
Upsell, for items with add ons. (e.g., "Would you like fries with that?")
"""

# Imports
import openai
from processResponse import processResponse

key = "" # OpenAI API key
openai.api_key = key
model_id = ""
debug_mode = False


# This is used by the app.py file, if you want to run in terminal, use debug_mode = True
def getBotResponse(prompt, actionHistory, orderHistory):
    response = openai.Completion.create(
        model=model_id,
        prompt=prompt,
        max_tokens=100, 
        n=1,  
        stop="###",  
    )
    response = response.choices[0].text.strip()
    botResponse = processResponse(response, actionHistory, orderHistory)
    return botResponse



if debug_mode:
    # The bot can have many actions like greet, add items to order, update order etc.
    actionHistory = []
    # the order can change every step of the way, it might be useful to keep a history
    orderHistory = [[]]
    while True:
        prompt = input("Your Input:\n")
        prompt += " ->"

        response = openai.Completion.create(
            model=model_id,
            prompt=prompt,
            max_tokens=100,  # Adjust the value as per your desired completion length
            n=1,  # Number of completions to generate
            stop="###",  # Specify a stopping condition if desired
        )
        response = response.choices[0].text.strip()
        print(response)
        botResponse = processResponse(response, actionHistory, orderHistory)
        print()
        print("RestaurantBot:\n" + botResponse)
        print()
        
        if botResponse == "Okay, have a nice day":
            break
