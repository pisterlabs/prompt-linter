import cohere
import json
co = cohere.Client('jY0mYl6MTtHnKLxMAPYBGg69rWrxt2CW9Kh0J1TJ')
from cohere.classify import Example

examples=[  Example("I want to have lunch", "eat out"),  Example("I want to have apple", "eat out"),  Example("I want to go to cafe", "eat out"),  Example("I want to have try new restaurants", "eat out"),  Example("I want to train", "gym buddy"),  Example("I want to exercise", "gym buddy"),  
          Example("I want to do cardio", "gym buddy"),Example("I want some to teach me", "Looking for mentor"), Example("I want to learn css", "Looking for mentor"), Example("I want to learn skiing", "Looking for mentor")
          
          ]
inputs=["I want to eat apple"]

response = co.classify(
  inputs=inputs,
  examples=examples,
)
classifications_dict = vars(response)
# Write the response to a local JSON file
with open('response.json', 'w') as f:
    json.dump(response, f)

