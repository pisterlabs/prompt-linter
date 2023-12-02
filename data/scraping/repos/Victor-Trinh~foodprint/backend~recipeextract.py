import json 
import cohere
import pandas as pd
import requests
import datetime
import os
import numpy as np
from tqdm import tqdm
import random

# https://github.com/cohere-ai/notebooks/blob/main/notebooks/Entity_Extrcation_with_Generative_Language_Models.ipynb
class cohereExtractor():
    def __init__(self, examples, example_labels, labels, task_desciption, example_prompt, co):
        self.examples = examples
        self.example_labels = example_labels
        self.labels = labels
        self.task_desciption = task_desciption
        self.example_prompt = example_prompt
        self.co = co

    def make_prompt(self, example):
        examples = self.examples + [example]
        labels = self.example_labels + [""]
        return (self.task_desciption +
                "\n---\n".join( [examples[i] + "\n" +
                                self.example_prompt + 
                                 labels[i] for i in range(len(examples))]))

    def extract(self, example):
      extraction = self.co.generate(
          model='df26095e-6137-4e4c-aea9-be0849745670-ft',
          prompt=self.make_prompt(example),
          max_tokens=10,
          temperature=0.1,
          stop_sequences=["\n"])
      return(extraction.generations[0].text[:-1])

# get random (ingredient, instructions)
def get_random_recipes(data, n, i = None):
  recipes = []
  if i:
    for _ in range(n):
      recipes.append(("; ".join([food['text'] for food in data[i]['annotation']]), # extracted
                      data[i]['infon'][1])) # instructions
    return recipes
  idx = list(range(n))
  #idx = np.random.randint(0, len(data), size = n)
  random.shuffle(idx)
  for id in idx:
    try:
      recipes.append(("; ".join([food['text'] for food in data[id]['annotation']]), # extracted
                      data[id]['infon'][1])) # instructions
    except:
      continue
  return recipes

# get random (instructions)
def get_random_instructions(data, n, i = None):
  recipes = []
  if i:
    for _ in range(n):
      recipes.append(data[i]['infon'][1]) # instructions
    return recipes
  idx = np.random.randint(0, len(data), size = n)
  for id in idx:
    recipes.append(data[id]['infon'][1]) # instructions
  return recipes

def food_extract(api_key, instructions):
  co = cohere.Client(api_key)

  # load recipe data https://academic.oup.com/database/article/doi/10.1093/database/baz121/5611291
  with open('../data/xmltojson.json') as fp:
      data = json.load(fp)
  data = data['collection']['document']
  data = list(data)

  # get recipes 
  recipe_examples = get_random_recipes(data, n = 4, i = 1) #get_random_recipes(data, 3) 
  #instructions = get_random_instructions(data, 1)#[recipe_examples[0][1]]
  cohereFoodExtractor = cohereExtractor([e[1] for e in recipe_examples],
                                        [e[0] for e in recipe_examples], 
                                        [],
                                        "",
                                        "extract food from recipe: ",
                                        co)
  #print(cohereFoodExtractor.make_prompt('<input text here>'))

  # extract food from recipes
  results = []
  for text in tqdm(instructions):
    try:
      extracted_text = cohereFoodExtractor.extract(text)
      results.append(extracted_text)
    except Exception as e:
      print('ERROR: ', e)

  #food_extractions = pd.DataFrame(data={'text': instructions, 'extracted_text': results})
  #process strings
  results = "".join(results).split(';')
  fin = []
  for food in results:
    fin.append(food.strip())
  #print("".join(instructions) + '\n' + str(fin))
  return fin

if __name__ == "__main__":
  from dotenv import load_dotenv
  load_dotenv()
  # co = cohere.Client(os.environ['COHERE_KEY'])

  # # load recipe data https://academic.oup.com/database/article/doi/10.1093/database/baz121/5611291
  # with open('data/xmltojson.json') as fp:
  #     data = json.load(fp)
  # data = data['collection']['document']
  # data = list(data)

  # # get recipes 
  # recipe_examples = get_random_recipes(data, 4)#[get_random_recipes(data, 1)[0] * 10]
  # instructions = get_random_instructions(data, 1)#[recipe_examples[0][1]]
  # cohereFoodExtractor = cohereExtractor([e[1] for e in recipe_examples],
  #                                       [e[0] for e in recipe_examples], 
  #                                       [],
  #                                       "",
  #                                       "extract food from recipe: ")
  # #print(cohereFoodExtractor.make_prompt('<input text here>'))

  # # extract food from recipes
  # results = []
  # for text in tqdm(instructions):
  #   try:
  #     extracted_text = cohereFoodExtractor.extract(text)
  #     results.append(extracted_text)
  #   except Exception as e:
  #     print('ERROR: ', e)

  # #food_extractions = pd.DataFrame(data={'text': instructions, 'extracted_text': results})
  # #process strings
  # results = "".join(results).split(';')
  # fin = []
  # for food in results:
  #   fin.append(food.strip())
  # print("".join(instructions) + '\n' + str(fin))
  print(food_extract(os.environ['COHERE_KEY'], ["Preheat oven to 275 degrees F (135 degrees C). In a shallow baking dish combine the artichoke hearts, mozzarella cheese, parmesan cheese and mayonnaise. Bake for 45 minutes, or until hot and bubbly. Sprinkle with almonds if desired. Serve hot with tortilla chips or crackers."]))