import re
import openai
from nltk.corpus import wordnet as wn
import json


#same as evalution.py but it reads json file instead of xml and gold key txt

openai.api_key = 'replace with openai api key'
sense_keys = []
ok = 0
notok = 0

def get_word_meanings(word):
  global sense_keys

  sense_keys.clear()

  synsets = wn.synsets(word)

  for i, synset in enumerate(synsets):
    sense_keys.append((i, synset.name()))
  return 0

def same_check(model_question, real_answer, keyword):
  response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=model_question,
    max_tokens=33,
    temperature=0,
    top_p=0.3,
    frequency_penalty=0,
    presence_penalty=0
  )
  get_word_meanings(keyword)
  reply = response.choices[0].text.strip()
  sense_key_num = re.search(r'\d+', reply)
  synset_id = None
  if sense_key_num:
      chosen_number = int(sense_key_num.group())
      if chosen_number < len(sense_keys):
          synset_id = sense_keys[chosen_number][1]
  print(reply)
  print(synset_id)
  print(real_answer)
  if synset_id == real_answer:
    return True
  else:
    return False


with open('custom_data.json', 'r') as file:
  data = json.load(file)

counter = 0
for item in data:
  question = item['question']
  answer = item['answer']
  word = item['word']
  if same_check(question, answer, word) == True:
    ok += 1
  else:
    notok += 1
  counter += 1
  if counter == 45:
    break

precision = ok/(ok + notok)
#recall is always 1 here since the model is just doing mcq
f1 = (2*precision*1)/(precision+1)

print("ok: " + str(ok))
print("notok: " + str(notok))
print("precision: " + str(precision))
print("f1 socre: " + str(f1))