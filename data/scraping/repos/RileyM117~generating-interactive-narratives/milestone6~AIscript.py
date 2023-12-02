!pip install --upgrade pip
!pip install openai
!pip install fantasynames
import os
import openai
import random
import fantasynames as names
import re
import random_character

%env OPENAI_API_KEY = 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

names_list = []
rand_names = []

i = 0
while i < 100:
  names_list.append(names.human('male'))
  i+=1

while i < 200:
  names_list.append(names.human('female'))
  i+=1

while i < 210:
  rand_names.append(random.choice(names_list))
  i+=1
  
locations = ['Forest Path','Camp','Tavern','Blacksmith\'s Shop','Ruins']
  
%%capture gpt_output --no-stderr
completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "Write me a medieval story with a plot twist in the format of a script. The only locations you can use are" + ' '.join(locations) +   "but you do not have to use all of them. The only names you can use are" + ' '.join(rand_names) + " but you do not have to use all of them."}
    ]
)
response = completion.choices[0].message.content
response.replace('\n',' :')


story = re.sub(r"\s+", " ", response).strip()
# seperate story into sentences
sentences = story.split(".")
for sentence in sentences:
  sentence = sentence.strip()
  
words = story.split()
caps = []
# Loop through each word in the list
for word in words:
    # Check if the word is in all caps
    if word.isupper():
      if word != 'I' and word != 'FADE' and word != 'IN:' and word != 'OUT.' and word != 'EXT.' and word != 'DAY' and word != 'NIGHT' and word != 'INT.' and word != 'A':
        caps.append(word)
        
for i in range(len(caps)):
    caps[i] = caps[i].replace('.', '')
for i in range(len(caps)):
    caps[i] = caps[i].replace(',', '')  
for i in range(len(caps)):
    caps[i] = caps[i].replace(')', '')
    
caps_no_dupes = []
for element in caps:
    if element not in caps_no_dupes:
        caps_no_dupes.append(element)
        
for i in range(len(rand_names)):
    rand_names[i] = rand_names[i].upper()
    
characters = []

for item1 in caps_no_dupes:
    for item2 in rand_names:
        if item1 in item2:
            characters.append(item2)
            
char_list = []
for element in characters:
    if element not in char_list:
        char_list.append(element)
  
char_objects = []
for i in char_list:
  char_objects.append(RandomCharacter(i).get_rand_male())
