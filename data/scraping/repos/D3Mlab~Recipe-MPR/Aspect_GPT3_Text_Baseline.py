from helper import *
import openai
import time
from tqdm import tqdm
import random
import numpy as np
import re
from string import whitespace
from sklearn.utils import shuffle
import requests
from requests.exceptions import ConnectionError

openai.api_key = 'API_KEY'

def create_prompt(prompt_prefix, aspect, options_list):
  query = "Aspect: " + aspect + " \n\n"
  options = "Options: \n"

  for i in range(len(options_list)):
    options += str(i) + ". " + options_list[i] + "\n"

  # ask for specific score format in text response
  prompt_suffix = '''
Please answer in a format that looks like this:
Option 0 Score: _
Option 1 Score: _
Option 2 Score: _
Option 3 Score: _
Option 4 Score: _ '''

  full_prompt = prompt_prefix + query + options + prompt_suffix
  return full_prompt

def create_fs_prefix(prompt_prefix, train_data,prompt_size):
  sample_data = random.sample(train_data, prompt_size)
  prompt_text = prompt_prefix

  for s in sample_data:
    for a in list(s["correctness_explanation"].keys()):
      query = "Aspect: " + a + " \n\n"
      options = "Options: \n"
      ans_ind = 0
      options_list = [val for val in s['options'].values()]
      # shuffle order of options in prompt
      random.shuffle(options_list)

      prompt_suffix = '\n'
      for i in range(len(options_list)):
        options += str(i) + ". " + options_list[i] + "\n"
        # assign correct option a score of 1 and all other options a score of 0
        if options_list[i] == s['options'][s['answer']]:
          ans_ind = i
          prompt_suffix += 'Option ' + str(i) + ' Score: ' + str(1) + '\n'
        else:
          prompt_suffix += 'Option ' + str(i) + ' Score: ' + str(0) + '\n'

      full_prompt = query + options + prompt_suffix + '\n'
      prompt_text += full_prompt
  
  return prompt_text

# gets response from API
def get_response(prompt):
  tries = 5
  while True:
    tries -= 1
    try:
      response = openai.Completion.create(
                  model="text-davinci-003",
                  prompt=prompt,
                  temperature=0,
                  max_tokens=60
                )
      break
    except ConnectionError as err:
      if tries == 0:
        raise err
      else:
        time.sleep(5)

  #print(response)
  return response.choices[0].text

def parse_score_text(query, responses):
  all_scores = []
  # scores for each aspect
  for text in responses:
    split_text = text.splitlines()
    split_text = [x for x in split_text if x not in whitespace]

    aspect_scores = []
    option_score_dict = {}
    # scores for each option
    for span in split_text:
      option_num = re.search('Option (\d+)', span)
      option_score = re.search('Score: (\d+)', span)

      if option_num == None or option_score == None:
        print("Invalid option num or score")
        continue

      option_num = int(option_num.group(1))
      option_score = float(option_score.group(1))

      option_score_dict.update({option_num:option_score})
    
    # check if the number of scores corresponds to the number of options
    if len(option_score_dict) != 5:
      print("Invalid scores. Query: {}".format(query))
      print(text)
      print(option_score_dict)
      invalid += 1
      continue
    
    # get list of scores for each option in order
    for i in range(5):
      aspect_scores.append(option_score_dict[i])

    all_scores.append(aspect_scores)      

  return all_scores

def aspect_gpt3_pred(train_data, test_data, agg_fcn, prompt_size=5, fewshot=False):
  # prompt instruction
  prefix = "Given the preference aspect and five options, generate a list of scores for how well each option satisfies the query: \n\n"
  if fewshot:
    prefix = create_fs_prefix(prefix, train_data, prompt_size)

  predictions = []
  N = len(test_data)
  for i in tqdm(range(0, N, 1)):
    # limit API requests per minute
    if (i + 1) % 10 == 0:
      time.sleep(60)

    sample = test_data[i]
    options_list = [val for val in sample['options'].values()]
    query = sample['query']
    correct_answer = sample['options'][sample['answer']]
    aspects = sample['correctness_explanation'].keys()

    responses = []
    for a in aspects:
      prompt = create_prompt(prefix, a, options_list)
      # get API response
      response = get_response(prompt)
      responses.append(response)

    # parse text scores into float scores
    scores = parse_score_text(query, responses)

    agg_scores = aggregate(scores, agg_fcn)
    agg_scores, options_shuffled = shuffle(agg_scores, options_list, random_state=0)
    args = np.argsort(agg_scores)
    predicted = options_shuffled[args[-1]]
    predictions.append(predicted)
  
  return predictions