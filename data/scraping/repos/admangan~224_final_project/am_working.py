# %%
from transformers import pipeline
pipe = pipeline("text-classification", model="roberta-base-openai-detector")
print(pipe("Hello world! Is this content AI-generated?"))  # [{'label': 'Real', 'score': 0.8036582469940186}]
# %%
#load in training_set_rel3.tsv
import pandas as pd
import numpy as np
import openai
import os
import dotenv
import json
import requests
from dotenv import load_dotenv
from textblob import TextBlob
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

df = pd.read_csv('training_set_rel3.tsv', sep='\t', encoding='ISO-8859-1')
df = df[df['essay_set'].isin([1,2,7,8])]
df = df.reset_index(drop=True)
# %%
native_prompts = {
    1:'More and more people use computers, but not everyone agrees that this benefits society. Those who support advances in technology believe that computers have a positive effect on people. They teach hand-eye coordination, give people the ability to learn about faraway places and people, and even allow people to talk online with other people. Others have different ideas. Some experts are concerned that people are spending too much time on their computers and less time exercising, enjoying nature, and interacting with family and friends. Write a letter to your local newspaper in which you state your opinion on the effects computers have on people. Persuade the readers to agree with you.',
    2: 'Write an essay in which you explain how the author builds an argument to persuade his/her audience that authors claim. In your essay, analyze how the author uses one or more of the features listed above (or features of your own choice) to strengthen the logic and persuasiveness of his/her argument. Be sure that your analysis focuses on the most relevant features of the passage.',
    7: 'Write about patience. Being patient means that you are understanding and tolerant. A patient person experience difficulties without complaining. Do only one of the following: write a story about a time when you were patient OR write a story about a time when someone you know was patient OR write a story in your own way about patience.',
    8: 'We all understand the benefits of laughter. For example, someone once said, “Laughter is the shortest distance between two people.” Many other people believe that laughter is an important part of any relationship. Tell a true story in which laughter was one element or part.',
}

non_native_prompts = {
    1: 'Do you agree or disagree with the following statement? It is better to have broad knowledge of many academic subjects than to specialize in one specific subject. Use specific reasons and examples to support your answer.',
    2: 'Do you agree or disagree with the following statement? Young people enjoy life more than older people do. Use specific reasons and examples to support your answer.',
    3: 'Do you agree or disagree with the following statement? Young people nowadays do not give enough time to helping their communities. Use specific reasons and examples to support your answer.',
    4: 'Do you agree or disagree with the following statement? Most advertisements make products seem much better than they really are. Use specific reasons and examples to support your answer.',
    5: 'Do you agree or disagree with the following statement? In twenty years, there will be fewer cars in use than there are today. Use reasons and examples to support your answer.',
    6: 'Do you agree or disagree with the following statement? The best way to travel is in a group led by a tour guide. Use reasons and examples to support your answer.',
    7: 'Do you agree or disagree with the following statement? It is more important for students to understand ideas and concepts than it is for them to learn facts. Use reasons and examples to support your answer.',
    8: 'Do you agree or disagree with the following statement? Successful people try new things and take risks rather than only doing what they already know how to do well. Use reasons and examples to support your answer.'
}

# %%

class GPTZeroAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = 'https://api.gptzero.me/v2/predict'
    def text_predict(self, document):
        url = f'{self.base_url}/text'
        headers = {
            'accept': 'application/json',
            'X-Api-Key': self.api_key,
            'Content-Type': 'application/json'
        }
        data = {
            'document': document
        }
        response = requests.post(url, headers=headers, json=data)
        return response.json()
    def file_predict(self, file_path):
        url = f'{self.base_url}/files'
        headers = {
            'accept': 'application/json',
            'X-Api-Key': self.api_key
        }
        files = {
            'files': (os.path.basename(file_path), open(file_path, 'rb'))
        }
        response = requests.post(url, headers=headers, files=files)
        return response.json()

#gptzero = GPTZeroAPI(Insert API Key Here)
# %%
def completion_3_5(prompt, temp=0.9): 
  comp = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "user",
            "content": "Write a 250 word essay responding to this prompt: " + prompt
         }
    ],
    temperature=temp
  )
  return comp.choices[0].message.content

def completion_4(prompt, temp=0.9):
  comp = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {
            "role": "user",
            "content": "Write a 250 word essay responding to this prompt: " + prompt
          }
    ],
    temperature=temp
  )
  return comp.choices[0].message.content

def completion_3(prompt, temp=0.9):
  comp = openai.Completion.create(
    engine="text-davinci-003",
    prompt="Write a 250 word essay responding to this prompt: " + prompt,
    temperature=temp,
    max_tokens=250
  )
  return comp.choices[0].text

def correct_spelling(text):
    text = TextBlob(text)
    result = text.correct()
    return result

def gpt_generation(output, prompts, native):
  for i in range(0, 4):
    print('Generating essay ' + str(i) + ' of 4')
    for prompt in prompts.values():
      for model in ['completion_3','completion_4', 'completion_3_5']:
        entry = {}
        entry['prompt'] = prompt
        entry['essay'] = eval(model)(prompt)
        entry['native'] = native
        output.append(entry)
  return output

def save_json(file_name, file_contents):
  with open(file_name, 'w') as f:
    json.dump(file_contents, f, indent=4)
# %%

# Organize the data
# We need two lists of gpt generated essays, one for the native and one for the non-native prompts
gpt_generated_native = []
gpt_generated_non_native = []

# %%
len(gpt_generated_native)
len(gpt_generated_non_native)
# %%

gpt_generation(gpt_generated_native, native_prompts, True)
save_json('gpt_generated_native.json', gpt_generated_native)

# %%
gpt_generation(gpt_generated_non_native, non_native_prompts, False)


save_json('gpt_generated_non_native.json', gpt_generated_non_native)



# %%

# We need one list of native essays, repeated three times, with original, spelling, and capitalization flags

# Get the NLI essays from txt and save them to a list
non_native_list = []

for filename in os.listdir('drive-download-20230601T034313Z-001'):
  with open('drive-download-20230601T034313Z-001/' + filename, 'r') as f:
    non_native_list.append(f.read())

non_native_essays = []

def non_native_essay_data_clean():
  non_native_essays = []
  for essay in non_native_list:
    entry = {}
    entry['native'] = False
    entry['generated'] = False
    entry['spelling'] = False
    entry['essays'] = essay
    non_native_essays.append(entry)
  save_json('non_native_essays_orig.json', non_native_essays)
  non_native_essays_spell = non_native_essays.copy()
  for essay in non_native_essays_spell:
    essay['spelling'] = True
    essay['essays'] = str(correct_spelling(essay['essays']))
  save_json('non_native_essays_spell.json', non_native_essays_spell)
  return non_native_essays, non_native_essays_spell

non_native_essays, non_native_essays_spell = non_native_essay_data_clean()
# We need one list of non-native essays, repeated three times, with original, spelling, and capitalization flags
# df = pd.read_csv('training_set_rel3.tsv', sep='\t', encoding='ISO-8859-1')
# df = df[df['essay_set'].isin([1,2,7,8])]
# df = df.reset_index(drop=True)


# %%

print(type(correct_spelling(non_native_list[0])))
# convert textblob to string

# %%

def native_essay_data_clean(df):
  native_essays = []
  for i in range(0, 50):
    entry = {}
    entry['native'] = True
    entry['generated'] = False
    entry['spelling'] = False
    entry['essays'] = df['essay'][i]
    native_essays.append(entry)
  save_json('native_essays_orig.json', native_essays)
  native_essays_spell = native_essays.copy()
  for essay in native_essays_spell:
    essay['spelling'] = True
    essay['essays'] = str(correct_spelling(essay['essays']))
  save_json('native_essays_spell.json', native_essays_spell)
  return native_essays, native_essays_spell

native_essays, native_essays_spell = native_essay_data_clean(df)

# %%
  

# %%
def gpt_zero_prediction(essays, results, generated):
  for i in range(len(essays)):
    entry = {}
    entry['native'] = essays[i]['native']
    entry['generated'] = generated
    try:
      entry['spelling'] = essays[i]['spelling']
    except:
      entry['spelling'] = False
    average_prob = gptzero.text_predict(essays[i]['essays'])['documents'][0]['average_generated_prob']
    entry['average_probability'] = average_prob
    results.append(entry)
  return results

# %%
gpt_generated_native = json.load(open('gpt_generated_native.json'))
gpt_generated_non_native = json.load(open('gpt_generated_non_native.json'))

for i in range(len(gpt_generated_native)):
  gpt_generated_native[i]['essays'] = gpt_generated_native[i]['essay']
  del gpt_generated_native[i]['essay']

for i in range(len(gpt_generated_non_native)):
  gpt_generated_non_native[i]['essays'] = gpt_generated_non_native[i]['essay']
  del gpt_generated_non_native[i]['essay']

# %%
results_normal = []
results_normal = gpt_zero_prediction(native_essays, results_normal, False)
results_normal = gpt_zero_prediction(non_native_essays, results_normal, False)

# %%
results_4 = []
results_4 = gpt_zero_prediction(gpt_generated_native, results_4, True)
results_4 = gpt_zero_prediction(gpt_generated_non_native, results_4, True)
print(len(results_4))

# %%
results_spell = []
results_spell = gpt_zero_prediction(native_essays_spell, results_spell, False)
# %%
print(results_spell[0])
# %%
results_2 = []
results_2 = gpt_zero_prediction(non_native_essays_spell, results_2, False)

# %%

results_3 = []
results_3 = gpt_zero_prediction(gpt_generated_native, results_3, True)
results_3 = gpt_zero_prediction
(gpt_generated_non_native, results_3, True)

#%%
len(results_spell)
# %%
save_json('results_spell.json', results_spell)

# %%

for i in range(25):
  entry = {}
  entry['native'] = False
  entry['generated'] = False
  average_prob = gptzero.text_predict(df['essay'][i])['documents'][0]['average_generated_prob']
  entry['average_probability'] = average_prob
  test_set.append(entry)
# %%
print(len(results_normal))

# %%
# remove 368 from results
native_results_spell = []
non_native_results_spell = []
for i in range(len(results_normal)):
  if results_normal[i]['native'] == True:
    native_results__no_spell.append(results_normal[i])
  else:
    non_native_results_no_spell.append(results_normal[i])


# %%
save_json('native_results_no_spell.json', native_results__no_spell)
save_json('non_native_results_no_spell.json', non_native_results_no_spell)
# %%
data_results_no_spell = []
# %%
#count the number of correct predictions
import operator

for i in range(41):
  threshold = 0.6 + i * 0.01
  test_set = results_normal.copy()
  for i in range(len(test_set)):
    if operator.ge(test_set[i]['average_probability'], threshold):
      test_set[i]['predicted_generated'] = True
    else:
      test_set[i]['predicted_generated'] = False

  true_positive = 0
  true_negative = 0
  false_positives = 0
  false_negatives = 0
  for i in range(len(test_set)):
    if test_set[i]['predicted_generated'] == test_set[i]['generated']:
      true_positive += 1
    elif test_set[i]['predicted_generated'] == True and test_set[i]['generated'] == False:
      false_positives += 1
    elif test_set[i]['predicted_generated'] == False and test_set[i]['generated'] == True:
      false_negatives += 1
    else:
      true_negative += 1


  result_entry = {}
  result_entry['threshold'] = threshold
  result_entry['true_positive'] = true_positive
  result_entry['true_negative'] = true_negative
  result_entry['false_positives'] = false_positives
  result_entry['false_negatives'] = false_negatives
  if result_entry not in data_results_no_spell:
    data_results_no_spell.append(result_entry)

  print(len(test_set))
  print(true_positive)
  print(true_negative)
  print(false_positives)
  print(false_negatives)

# %%
save_json('data_results_no_spell.json', data_results_no_spell)
# %%

# %%
# load results from json
non_native_data_results = json.load(open('non_native_data_results.json'))

native_data_results = json.load(open('native_data_results.json'))

overall_results = json.load(open('data_results.json'))

# %%


# %%
# calculate the f1 score for each class of essay. Native, non-native, generated
# f1 score = 2 * (precision * recall) / (precision + recall)
# precision = true positives / (true positives + false positives)
# recall = true positives / (true positives + false negatives)
# accuracy = (true positives + true negatives) / (true positives + true negatives + false positives + false negatives)

precision_native = []
recall_native = []
f1_native = []
accuracy_native = []
for i in range(len(native_data_results)):
  precision_native.append(native_data_results[i]['correct'] / (native_data_results[i]['correct'] + native_data_results[i]['false_positives']))
  recall_native.append(native_data_results[i]['correct'] / (native_data_results[i]['correct'] + native_data_results[i]['false_negatives']))
  f1_native.append(2 * (precision_native[i] * recall_native[i]) / (precision_native[i] + recall_native[i]))
  accuracy_native.append((native_data_results[i]['correct'] + native_data_results[i]['false_negatives']) / (native_data_results[i]['correct'] + native_data_results[i]['false_positives'] + native_data_results[i]['false_negatives'] + native_data_results[i]['correct']))

average_precision_native = sum(precision_native) / len(precision_native)
average_recall_native = sum(recall_native) / len(recall_native)
average_f1_native = sum(f1_native) / len(f1_native)
average_accuracy_native = sum(accuracy_native) / len(accuracy_native)
print("Native Precision: ", average_precision_native)
print("Native Recall: ", average_recall_native)
print("Native F1: ", average_f1_native)
print("Native Accuracy: ", average_accuracy_native)


precision_non_native = []
recall_non_native = []
f1_non_native = []
accuracy_non_native = []
for i in range(len(non_native_data_results)):
  precision_non_native.append(non_native_data_results[i]['correct'] / (non_native_data_results[i]['correct'] + non_native_data_results[i]['false_positives']))
  recall_non_native.append(non_native_data_results[i]['correct'] / (non_native_data_results[i]['correct'] + non_native_data_results[i]['false_negatives']))
  f1_non_native.append(2 * (precision_non_native[i] * recall_non_native[i]) / (precision_non_native[i] + recall_non_native[i]))
  accuracy_non_native.append((non_native_data_results[i]['correct'] + non_native_data_results[i]['false_negatives']) / (non_native_data_results[i]['correct'] + non_native_data_results[i]['false_positives'] + non_native_data_results[i]['false_negatives'] + non_native_data_results[i]['correct']))

average_non_native_precision = sum(precision_non_native) / len(precision_non_native)
average_non_native_recall = sum(recall_non_native) / len(recall_non_native)
average_non_native_f1 = sum(f1_non_native) / len(f1_non_native)
average_non_native_accuracy = sum(accuracy_non_native) / len(accuracy_non_native)

print("Non-Native Precision: ", average_non_native_precision)
print("Non-Native Recall: ", average_non_native_recall)
print("Non-Native F1: ", average_non_native_f1)
print("Non-Native Accuracy: ", average_non_native_accuracy)



# %%
native_data_results_no_spell = json.load(open('native_data_results_no_spell.json'))
native_data_results_spell = json.load(open('native_data_results_spell.json'))

non_native_data_results_no_spell = json.load(open('non_native_data_results_no_spell.json'))
non_native_data_results_spell = json.load(open('non_native_data_results_spell.json'))

data_results_no_spell = json.load(open('data_results_no_spell.json'))
data_results_spell = json.load(open('data_results_spell.json'))

# %%

# create a function that takes in a list of results and returns the average precision, recall, f1 and accuracy
def calculate_results(results):
  precision = []
  recall = []
  f1 = []
  accuracy = []
  for i in range(len(results)):
    precision.append(results[i]['true_positive'] / (results[i]['true_positive'] + results[i]['false_positives']))
    recall.append(results[i]['true_positive'] / (results[i]['true_positive'] + results[i]['false_negatives']))
    f1.append(2 * (precision[i] * recall[i]) / (precision[i] + recall[i]))
    accuracy.append((results[i]['true_positive'] + results[i]['true_negative']) / (results[i]['true_positive'] + results[i]['true_negative'] + results[i]['false_positives'] + results[i]['false_negatives']))

  average_precision = sum(precision) / len(precision)
  average_recall = sum(recall) / len(recall)
  average_f1 = sum(f1) / len(f1)
  average_accuracy = sum(accuracy) / len(accuracy)
  return average_precision, average_recall, average_f1, average_accuracy

# %%

# for each of the 6 lists (native_data_results_no_spell, native_data_results_spell, non_native_data_results_no_spell, non_native_data_results_spell, data_results_no_spell, data_results_spell) calculate the average precision, recall, f1 and accuracy and store each as an object in a list labeled
results = []
entry = {}
entry['name'] = 'native_data_results_no_spell'
entry['precision'], entry['recall'], entry['f1'], entry['accuracy'] = calculate_results(native_data_results_no_spell)
results.append(entry)

entry = {}
entry['name'] = 'native_data_results_spell'
entry['precision'], entry['recall'], entry['f1'], entry['accuracy'] = calculate_results(native_data_results_spell)
results.append(entry)

entry = {}
entry['name'] = 'non_native_data_results_no_spell'
entry['precision'], entry['recall'], entry['f1'], entry['accuracy'] = calculate_results(non_native_data_results_no_spell)
results.append(entry)

entry = {}
entry['name'] = 'non_native_data_results_spell'
entry['precision'], entry['recall'], entry['f1'], entry['accuracy'] = calculate_results(non_native_data_results_spell)
results.append(entry)

entry = {}
entry['name'] = 'data_results_no_spell'
entry['precision'], entry['recall'], entry['f1'], entry['accuracy'] = calculate_results(data_results_no_spell)
results.append(entry)

entry = {}
entry['name'] = 'data_results_spell'
entry['precision'], entry['recall'], entry['f1'], entry['accuracy'] = calculate_results(data_results_spell)
results.append(entry)



# %%
# save as a json file
with open('data_statistics_final.json', 'w') as outfile:
    json.dump(results, outfile, indent=4)
# %%
#plot the rate of false positives and false negatives for each of the 6 lists as a percentage of the total number of samples with the percentage on the y access and the threshold on the x access

import matplotlib.pyplot as plt

x = []
y = []
for i in range(len(native_data_results_no_spell)):
  x.append(native_data_results_no_spell[i]['threshold'])
  y.append(native_data_results_no_spell[i]['false_positives'] / (native_data_results_no_spell[i]['false_positives'] + native_data_results_no_spell[i]['false_negatives']))

plt.plot(x, y, label='native_data_results_no_spell')

# %%
# do it for the overal data
x = []
y = []
for i in range(len(data_results_no_spell)):
  x.append(data_results_no_spell[i]['threshold'])
  y.append(data_results_no_spell[i]['false_positives'] / (data_results_no_spell[i]['false_positives'] + data_results_no_spell[i]['false_negatives']))
# label the axes
plt.xlabel('GPTZero Average Probability Threshold')
plt.ylabel('Percentage of False Positives')
plt.plot(x, y, label='data_results_no_spell')

# %%
