#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
[env]
# Conda Environment
# NO CONDA ENV

conda create --name ai_chatgpt_prompts python=3.9.13
conda info --envs
source activate ai_chatgpt_prompts
conda deactivate
# if needed to remove
conda env remove -n [NAME_OF_THE_CONDA_ENVIRONMENT]


# update conda 
conda update -n base -c defaults conda

[filename]
018_project_1_python_documentation_default_explain_code_code_chatgpt_api.py

[path]
cd /Users/brunoflaven/Documents/03_git/BlogArticlesExamples/ai_chatgpt_prompts/project_1_python_documentation_chatgpt_api/


[run]
python 018_project_1_python_documentation_default_explain_code_code_chatgpt_api.py

pip install Random

+ GREAT EXAMPLE FROM OPENAI.COM
https://platform.openai.com/examples







"""

import os
import openai

# personal configuration
import config_values.values_conf as conf

OPENAI_ORGANIZATION = conf.OPENAI_ORGANIZATION
OPENAI_API_KEY = conf.OPENAI_API_KEY

# quick and dirty
openai.organization = OPENAI_ORGANIZATION
# PAID ONE DO NOT DISPLAY
openai.api_key = OPENAI_API_KEY


response = openai.Completion.create(
    model="text-davinci-003",
    
    # prompt="class Log:\n    def __init__(self, path):\n        dirname = os.path.dirname(path)\n        os.makedirs(dirname, exist_ok=True)\n        f = open(path, \"a+\")\n\n        # Check that the file is newline-terminated\n        size = os.path.getsize(path)\n        if size > 0:\n            f.seek(size - 1)\n            end = f.read(1)\n            if end != \"\\n\":\n                f.write(\"\\n\")\n        self.f = f\n        self.path = path\n\n    def log(self, event):\n        event[\"_event_id\"] = str(uuid.uuid4())\n        json.dump(event, self.f)\n        self.f.write(\"\\n\")\n\n    def state(self):\n        state = {\"complete\": set(), \"last\": None}\n        for line in open(self.path):\n            event = json.loads(line)\n            if event[\"type\"] == \"submit\" and event[\"success\"]:\n                state[\"complete\"].add(event[\"id\"])\n                state[\"last\"] = event\n        return state\n\n\"\"\"\nHere's what the above class is doing, explained in a concise way:\n1.",
    
    prompt=" from bs4 import BeautifulSoup\n\n with open('example_3.html', 'r') as file: \n html=file.read()\n soup=BeautifulSoup(html, 'html.parser')\n    smallBookTitles=[h5.text for h5 in soup.find_all('h5')]\n    bookTitles=[]\n    for h6 in soup.find_all('h6'): \n    p=h6.find_next('p')\n    bookTitles.append(p.text)\n    print('Small book titles:', smallBookTitles)\n print('All book titles:', bookTitles)\"\"\"\nHere's what the above class is doing, explained in a concise way:\n1.",
    
    temperature=0,
    max_tokens=150,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=["\"\"\""]
)

print('\n --- RESULT')
# print(response)
# print(response.choices)
desired_text = response.choices[0].text
print(desired_text)


"""
-- FUNCTION
from bs4 import BeautifulSoup
with open('example_3.html', 'r') as file:
    html = file.read()

soup = BeautifulSoup(html, 'html.parser')
smallBookTitles = [h5.text for h5 in soup.find_all('h5')]
bookTitles = []
for h6 in soup.find_all('h6'):
    # Find the next <p> tag after the <h6> tag and extract its text content
    p = h6.find_next('p')
    bookTitles.append(p.text)

print('Small book titles:', smallBookTitles)
print('All book titles:', bookTitles)

-- RESULT
It opens the example_3.html file and reads the HTML code.
2. It creates a BeautifulSoup object from the HTML code.
3. It finds all the <h5> tags in the HTML code and stores the text from each tag in a list called smallBookTitles.
4. It finds all the <h6> tags in the HTML code and then finds the next <p> tag after each <h6> tag. It stores the text from each <p> tag in a list called bookTitles.
5. It prints out the smallBookTitles and bookTitles lists.


"""